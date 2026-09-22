"""Frozen ownership, exact per-domain CE exposure and paired report contracts."""
import copy
import json
import math
from types import SimpleNamespace

import pytest

from scripts import olmo_o5c_report as report


def container(rows=512, *, improvement=0., records=True, pass0_shift=0.):
    passes = []
    for index in (0, 1):
        values = []
        for i in range(rows):
            count = 80 + i % 7
            mean = 2. + .01 * (i % 11) + (1 + improvement if index else pass0_shift)
            values.append({"batch_index": i // 8, "row_index": i % 8, "document_id": i // 3,
                "ce_count": count, "ce_sum": mean * count, "mean_nll": mean, "next_token_correct": count // 2})
        count = sum(v["ce_count"] for v in values)
        total = math.fsum(v["ce_sum"] for v in values)
        hits = sum(v["next_token_correct"] for v in values)
        result = {"schema": "olmo-fbt-pass-evaluation-v1", "execution": "finite", "pass_index": index,
            "precision": "bf16_mixed", "mode": report.fbt._mode(1.), "positions_per_chunk": 128,
            "batches": math.ceil(rows / 8), "documents": rows, "input_tokens": count + rows,
            "ce_count": count, "ce_sum": total, "mean_nll": total / count, "perplexity": math.exp(total / count),
            "next_token_correct": hits, "next_token_accuracy": hits / count}
        if records: result["document_records"] = values
        passes.append(result)
    keys = ("precision", "mode", "execution", "positions_per_chunk", "batches", "documents", "input_tokens")
    return {"schema": "olmo-fbt-evaluation-v1", **{k: passes[0][k] for k in keys}, "passes": passes}


def domain_counts(arm, update):
    quotas = (16, 0) if arm == "code" else (8, 8)
    segments = (2, 0) if arm == "code" else (1, 2)
    return {domain: {"ce_positions": quota * update, "documents": rows * update,
                     "input_tokens": (quota + rows) * update}
            for domain, quota, rows in zip(report.DOMAINS, quotas, segments)}


def plan(arm):
    snapshots = [domain_counts(arm, update) for update in range(5)]
    return {"arm": arm, "total_updates": 4, "ce_per_update": 16,
        "batch_ce_prefix": [16 * update for update in range(5)],
        "batch_token_prefix": [sum(v["input_tokens"] for v in counts.values()) for counts in snapshots],
        "batch_row_prefix": [sum(v["documents"] for v in counts.values()) for counts in snapshots],
        "domain_prefixes": {domain: {key: [counts[domain][key] for counts in snapshots]
                            for key in ("input_tokens", "ce_positions", "documents")} for domain in report.DOMAINS}}


@pytest.fixture
def inputs():
    sources = {"scripts/olmo_o5c_train.py": "b" * 64, "cdrm/pretrained/olmo_fbt.py": "c" * 64}
    frozen_names = {"backbone.backbone.transformer.wte.weight", "backbone.fusion.output_scale"}
    frozen_names.update(f"backbone.backbone.transformer.blocks.{i}.{n}.weight" for i in range(2)
                        for n in ("att_proj", "attn_out", "ff_proj", "ff_out"))
    frozen = {name: "f" * 64 for name in frozen_names}
    layout = [{"name": name, "shape": [32, 32], "numel": 1024} for name in sorted(report.TRAINABLE_NAMES)]
    schedule = {"total_updates": 4, "ce_per_update": 16, "total_ce": 64, "arms": {arm: plan(arm) for arm in report.ARMS}}
    config = {"schema": "olmo-o5c-pilot-config-v1", "source_hashes": sources, "schedule": schedule,
        "source_checkpoint_sha256": "e" * 64, "data_manifest_sha256": "d" * 64,
        "physical_batch_size": 2, "precision": "bf16_mixed", "eval_rows": 128, "final_eval_rows": 512,
        "num_passes": 2, "gamma": 1., "beta": 1., "rt_layers": [], "nextlat_enabled": False,
        "prefix_mixin": False, "hidden_jitter": 0., "native_backbone_frozen": True,
        "model_config": {"model_dim": 32, "num_layers": 2}, "frozen_state_initial": frozen, "trainable_layout": layout}
    runtime = {"gpu": "H100", "cuda": "test", "torch": "test"}
    endpoint = {"checkpoint_sha256": "e" * 64, "frozen_state_digests": frozen, "trainable_parameters": layout,
        "state_digests": {**frozen, **{name: "a" * 64 for name in report.TRAINABLE_NAMES}},
        "optimizer_state": "fresh; endpoint optimizer/scheduler/counters/RNG not resumed"}
    preflight = {"schema": "olmo-o5c-preflight-v1", "status": "passed", "finished_utc": "now",
        "source_hashes": sources, "schedule": schedule, "source_checkpoint_sha256": "e" * 64,
        "data_manifest_sha256": "d" * 64, "frozen_state_initial": frozen, "trainable_layout": layout, "runtime": runtime, "endpoint": endpoint,
        "baseline": {split: container() for split in report.SPLITS},
        "baseline_small": {split: container(128, records=False) for split in report.SPLITS}}
    arms = {}
    for arm in report.ARMS:
        events = []
        for update in range(5):
            counts = domain_counts(arm, update)
            delta = (-.2 if arm == "code" else -.4) * update / 4
            events.append({"update": update, "input_tokens": sum(v["input_tokens"] for v in counts.values()),
                "ce_positions": update * 16, "domain_counts": counts, "beta": 1., "full": False, "rows_requested": 128,
                "metrics": {split: container(128, improvement=delta, records=False) for split in report.SPLITS}})
        final = {**events[-1], "full": True, "rows_requested": 512,
                 "metrics": {split: container(improvement=delta) for split in report.SPLITS}}
        counters = {"optimizer_updates": 4, "microbatches": 4 if arm == "code" else 8,
            "documents": sum(v["documents"] for v in counts.values()), "input_tokens": final["input_tokens"],
            "ce_positions": 64, "latent_pairs": 0, "kl_triples": 0}
        prefix = "gs://fast-chunks/test"
        checkpoint = {"optimizer_updates": 4, "input_tokens": final["input_tokens"], "size_bytes": 1000, "sha256": "a" * 64,
            "storage": {"uri": prefix + f"/{arm}/update-000004.pt", "generation": "123", "size_bytes": 1000,
                "sha256": "a" * 64, "md5_base64": "test", "verification": "verified"}}
        arms[arm] = {"schema": "olmo-o5c-arm-v1", "status": "completed", "arm": arm, "finished_utc": "now",
            "started_utc": "then", "configuration": config, "storage_prefix": prefix, "endpoint": endpoint,
            "source_fingerprint": {"checkpoint_sha256": "e" * 64, "source_checkpoint_sha256": "e" * 64,
                "data_manifest_sha256": "d" * 64, "code": sources, "runtime": runtime},
            "counters": counters, "data_cursor": 4, "domain_counts": counts, "checkpoints": [checkpoint],
            "frozen_state_initial": frozen, "frozen_state_final": frozen, "trainable_layout": layout,
            "evaluations": [*events, final], "wandb": {"run_url": "https://wandb.ai/test/project/runs/" + arm}}
    return json.loads(json.dumps([preflight, config, arms]))


def test_matched_ce_with_unequal_input_context_and_frozen_native_state(inputs):
    before = copy.deepcopy(inputs)
    result = report.build_comparison(*inputs)
    assert inputs == before
    assert result["matched_ce_exposure"] == 64
    assert result["arms"]["code"]["counters"]["input_tokens"] == 72
    assert result["arms"]["mixed"]["counters"]["input_tokens"] == 76
    assert result["arms"]["mixed"]["domain_counts"]["code"]["ce_positions"] == 32
    assert result["comparisons"]["retention_dev"]["mixed_pass1_minus_code_pass1"]["estimate"] == pytest.approx(-.2)
    assert result["comparisons"]["retention_dev"]["versus_source"]["mixed"]["estimate"] == pytest.approx(-.4)
    assert result["comparisons"]["dev"]["feedback_gap"]["code"]["estimate"] == pytest.approx(.8)
    assert all(e["rows_requested"] == 128 for rows in result["curves"].values() for e in rows)


@pytest.mark.parametrize("case", ["missing_arm", "running", "source", "runtime", "data", "source_endpoint", "config",
    "frozen_changed", "frozen_omitted", "scale_omitted", "foreign_frozen", "trainable_extra", "trainable_shape", "unfrozen",
    "beta", "gamma", "nextlat", "rt", "jitter", "quota", "prefix", "domain_prefix", "domain_total", "domain_balance",
    "counter", "cursor", "microbatches", "aux", "curve_domain", "curve_duplicate", "curve_initial", "curve_sample",
    "final_missing", "final_update", "final_sample", "pass0_changed", "pass_mode", "document_alignment", "checkpoint", "receipt",
    "fresh_optimizer", "fusion_start"])
def test_rejects_incompatible_or_incomplete_scientific_records(inputs, case):
    preflight, config, arms = inputs
    arm = arms["mixed"]
    final = arm["evaluations"][-1]
    if case == "missing_arm": del arms["code"]
    elif case == "running": arm["status"] = "running"
    elif case == "source": arm["source_fingerprint"]["code"]["scripts/olmo_o5c_train.py"] = "a" * 64
    elif case == "runtime": arm["source_fingerprint"]["runtime"]["torch"] = "other"
    elif case == "data": arm["source_fingerprint"]["data_manifest_sha256"] = "a" * 64
    elif case == "source_endpoint": arm["source_fingerprint"]["source_checkpoint_sha256"] = "a" * 64
    elif case == "config": arm["configuration"]["precision"] = "fp32"
    elif case == "frozen_changed": arm["frozen_state_final"]["backbone.backbone.transformer.wte.weight"] = "a" * 64
    elif case == "frozen_omitted":
        for obj in (preflight, config): del obj["frozen_state_initial"]["backbone.backbone.transformer.wte.weight"]
    elif case == "scale_omitted": del preflight["frozen_state_initial"]["backbone.fusion.output_scale"]
    elif case == "foreign_frozen": arm["frozen_state_initial"]["backbone.backbone.transformer.wte.weight"] = "a" * 64
    elif case == "trainable_extra": preflight["trainable_layout"].append({"name": "backbone.weight", "shape": [32, 32], "numel": 1024})
    elif case == "trainable_shape": preflight["trainable_layout"][0]["numel"] = 32
    elif case == "unfrozen": config["native_backbone_frozen"] = False
    elif case == "beta": config["beta"] = .5
    elif case == "gamma": config["gamma"] = .5
    elif case == "nextlat": config["nextlat_enabled"] = True
    elif case == "rt": config["rt_layers"] = [0]
    elif case == "jitter": config["hidden_jitter"] = .02
    elif case == "quota": config["schedule"]["total_ce"] = 65
    elif case == "prefix": config["schedule"]["arms"]["mixed"]["batch_token_prefix"][2] += 1
    elif case == "domain_prefix": config["schedule"]["arms"]["mixed"]["domain_prefixes"]["general"]["ce_positions"][2] -= 1
    elif case == "domain_total": arm["domain_counts"]["code"]["input_tokens"] += 1
    elif case == "domain_balance": arm["domain_counts"]["code"]["ce_positions"] += 1
    elif case == "counter": arm["counters"]["input_tokens"] -= 1
    elif case == "cursor": arm["data_cursor"] -= 1
    elif case == "microbatches": arm["counters"]["microbatches"] -= 1
    elif case == "aux": arm["counters"]["latent_pairs"] = 1
    elif case == "curve_domain": arm["evaluations"][1]["domain_counts"]["general"]["input_tokens"] += 1
    elif case == "curve_duplicate": arm["evaluations"].append(copy.deepcopy(arm["evaluations"][0]))
    elif case == "curve_initial": arm["evaluations"].pop(0)
    elif case == "curve_sample": arm["evaluations"][1]["rows_requested"] = 512
    elif case == "final_missing": arm["evaluations"].pop()
    elif case == "final_update": final["update"] -= 1
    elif case == "final_sample": final["rows_requested"] = 128
    elif case == "pass0_changed": final["metrics"]["dev"] = container(improvement=-.4, pass0_shift=.1)
    elif case == "pass_mode": final["metrics"]["dev"]["passes"][1]["mode"]["beta"] = .5
    elif case == "document_alignment": final["metrics"]["dev"]["passes"][1]["document_records"][0]["document_id"] = 999
    elif case == "checkpoint": arm["checkpoints"] = []
    elif case == "receipt": arm["checkpoints"][0]["storage"]["sha256"] = "b" * 64
    elif case == "fresh_optimizer": preflight["endpoint"]["optimizer_state"] = "resumed"
    elif case == "fusion_start": arm["endpoint"]["state_digests"]["backbone.fusion.state_proj.weight"] = "b" * 64
    with pytest.raises(ValueError): report.build_comparison(*inputs)


def test_cluster_resampling_combines_windows_and_is_deterministic():
    left = container(6, improvement=-.2)["passes"][1]
    right = container(6)["passes"][1]
    result = report.fbt.contrast(left, right)
    assert result["document_clusters"] == 2
    assert result["ci95"] == pytest.approx([-.2, -.2])
    assert result == report.fbt.contrast(left, right)


def write_inputs(tmp_path, inputs):
    preflight, config, arms = inputs
    directory = tmp_path / "preflight"; directory.mkdir()
    (directory / "report.json").write_text(json.dumps(preflight))
    (directory / "configuration.json").write_text(json.dumps(config))
    runs = tmp_path / "runs"
    for arm, value in arms.items():
        (runs / arm).mkdir(parents=True)
        (runs / arm / "report.json").write_text(json.dumps(value))
    return SimpleNamespace(preflight=directory, runs=runs, output_dir=tmp_path / "report")


def test_incomplete_queue_never_generates_results(tmp_path, inputs):
    inputs[2]["mixed"]["status"] = "running"
    args = write_inputs(tmp_path, inputs)
    with pytest.raises(ValueError, match="not completed"): report.build_report(args)
    assert not args.output_dir.exists()


def test_report_writes_standalone_figures_and_preserves_protocol(tmp_path, inputs):
    args = write_inputs(tmp_path, inputs)
    args.output_dir.mkdir(); protocol = args.output_dir / "protocol.md"; protocol.write_text("Frozen protocol\n")
    validated = report.validate_runs(args.preflight, args.runs)
    value = report.build_report(args)
    assert all(value[name] == expected for name, expected in validated.items())
    assert protocol.read_text() == "Frozen protocol\n"
    for name in (*report.FIGURES, "results.md", "final-comparison.json"):
        assert (args.output_dir / name).stat().st_size > 0
    assert (args.output_dir / "domain-exposure.pdf").read_bytes().startswith(b"%PDF")
    text = (args.output_dir / "results.md").read_text()
    assert "CE targets" in text and "not exact input tokens" in text and "half the code" in text
    assert "reserved tests untouched" in text and "not exact-online" in text
    assert value["report_source_sha256"] == report.sha(report.__file__)
