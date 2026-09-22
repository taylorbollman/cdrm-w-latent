"""Equal-data ordinary control reporting and historical endpoint alignment."""
import copy
import json
import math

import pytest

from scripts import olmo_o5e_report as report


def container(rows=512, *, kind="ordinary", shift=0., ordinary_shift=0., records=True):
    if kind == "ordinary":
        mode = {"enabled": False, "num_passes": 1, "beta": 0., "rt_mode": {"selected_layers": [], "alpha": 1.}}
    elif kind == "online":
        mode = {"beta": 1., "rt_mode": {"selected_layers": [], "alpha": 1.}}
    else:
        mode = {"enabled": True, "num_passes": 2, "beta": 1., "rt_mode": {"selected_layers": [], "alpha": 1.}}
    result = []
    for index in range(2 if kind == "fusion" else 1):
        delta = ordinary_shift if kind == "fusion" and index == 0 else shift
        values = []
        for i in range(rows):
            count = 100 + i % 7
            nll = 2. + .01 * (i % 11) + delta
            values.append({"batch_index": i // 8, "row_index": i % 8, "document_id": i // 3,
                "ce_count": count, "ce_sum": nll * count, "mean_nll": nll, "next_token_correct": count // 2})
        count = sum(v["ce_count"] for v in values)
        total = math.fsum(v["ce_sum"] for v in values)
        hits = sum(v["next_token_correct"] for v in values)
        metric = {"schema": "olmo-fbt-pass-evaluation-v1", "execution": "online" if kind == "online" else "finite",
            "pass_index": None if kind == "online" else index, "mode": mode, "precision": "bf16_mixed",
            "positions_per_chunk": 128, "documents": rows, "batches": math.ceil(rows / 8), "input_tokens": count + rows,
            "ce_count": count, "ce_sum": total, "mean_nll": total / count, "perplexity": math.exp(total / count),
            "next_token_correct": hits, "next_token_accuracy": hits / count}
        if records: metric["document_records"] = values
        result.append(metric)
    keys = ("mode", "precision", "execution", "positions_per_chunk", "documents", "batches", "input_tokens")
    return {"schema": "olmo-fbt-evaluation-v1", **{k: result[0][k] for k in keys}, "passes": result}


def domain_counts(update):
    return {domain: {"ce_positions": 4096 * update, "input_tokens": 4128 * update, "documents": 32 * update}
            for domain in ("code", "general")}


@pytest.fixture
def inputs(monkeypatch):
    # O5d's complete-schema validation has its own unchanged 34-test suite.
    # This fixture supplies its relevant validated endpoint records only.
    monkeypatch.setattr(report.online_report, "validate_report", lambda value: value)
    model = {"model_dim": 8, "mlp_intermediate_size": 16, "num_layers": 1, "vocab_size": 32}
    names = {"backbone.backbone.transformer.wte.weight": [32, 8],
        "backbone.backbone.transformer.blocks.0.att_proj.weight": [24, 8],
        "backbone.backbone.transformer.blocks.0.attn_out.weight": [8, 8],
        "backbone.backbone.transformer.blocks.0.ff_proj.weight": [32, 8],
        "backbone.backbone.transformer.blocks.0.ff_out.weight": [8, 16]}
    layout = [{"name": k, "shape": shape, "numel": math.prod(shape)} for k, shape in names.items()]
    initial = {k: "a" * 64 for k in set(names) | report.FROZEN_NAMES}
    frozen = {k: initial[k] for k in report.FROZEN_NAMES}
    plan = {"total_updates": 512, "ce_per_update": 8192,
        "batch_ce_prefix": [8192 * u for u in range(513)], "batch_token_prefix": [8256 * u for u in range(513)],
        "batch_row_prefix": [64 * u for u in range(513)],
        "domain_prefixes": {domain: {key: [domain_counts(u)[domain][key] for u in range(513)]
                            for key in ("ce_positions", "input_tokens", "documents")} for domain in ("code", "general")}}
    source_hashes = {"cdrm/pretrained/olmo_fbt.py": "b" * 64}
    base = {"model_config": model, "source_checkpoint_sha256": "e" * 64, "data_manifest_sha256": "f" * 64,
        "base_data_manifest_sha256": "d" * 64, "precision": "bf16_mixed", "attention_backend": "sdpa",
        "eval_batch_size": 8, "physical_batch_size": 16, "seed": 20260922, "eval_rows": 128, "final_eval_rows": 512}
    schedule = {"total_updates": 512, "ce_per_update": 8192, "total_ce": 4194304, "arms": {"mixed": plan}}
    config = {**base, "schema": "olmo-o5e-config-v1", "schedule": schedule, "data_plan_arm": "mixed",
        "fbt_enabled": False, "beta": 0., "num_passes": 1, "gamma": 1., "rt_layers": [], "nextlat_enabled": False,
        "native_backbone_frozen": False, "objective": "single_ordinary_ce", "lr": 1e-5, "warmup_updates": 50,
        "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1, "max_grad_norm": 1.,
        "trainable_layout": layout, "full_state_initial": initial, "frozen_state_initial": frozen,
        "source_hashes": {**source_hashes, "scripts/olmo_o5e_train.py": "c" * 64}}
    runtime = {"device": "H100"}
    preflight = {"schema": "olmo-o5e-preflight-v1", "status": "passed", "finished_utc": "now",
        "runtime": runtime, "source_hashes": config["source_hashes"], "full_state_initial": initial,
        "frozen_state_initial": frozen, "trainable_layout": layout,
        "baseline": {s: container() for s in report.SPLITS},
        "baseline_small": {s: container(128, records=False) for s in report.SPLITS}}
    events, fusion_curves = [], []
    for u in [0, 50, *range(64, 513, 64)]:
        event = {"update": u, "full": False, "rows_requested": 128, "beta": 0.,
            "input_tokens": 8256 * u, "ce_positions": 8192 * u, "domain_counts": domain_counts(u),
            "metrics": {s: container(128, shift=-.2 * u / 512, records=False) for s in report.SPLITS}}
        events.append(event)
        fusion_curves.append({**event, "beta": 1., "metrics": {s: container(128, kind="fusion", shift=1. - 1.1 * u / 512, records=False)
                                                              for s in report.SPLITS}})
    final = {**events[-1], "full": True, "rows_requested": 512,
             "metrics": {s: container(shift=-.2) for s in report.SPLITS}}
    prefix = "gs://fast-chunks/o5e-test"
    checkpoint = {"sha256": "9" * 64, "size_bytes": 1000, "optimizer_updates": 512, "input_tokens": 8256 * 512,
        "storage": {"uri": prefix + "/ordinary/update-000512.pt", "sha256": "9" * 64, "size_bytes": 1000,
                    "generation": "123", "md5_base64": "test", "verification": "verified"}}
    ordinary = {"schema": "olmo-o5e-arm-v1", "status": "completed", "arm": "ordinary", "finished_utc": "now", "started_utc": "then",
        "configuration": config, "full_state_initial": initial, "full_state_final": {**{k: "8" * 64 for k in names}, **frozen},
        "frozen_state_initial": frozen, "frozen_state_final": frozen, "trainable_layout": layout,
        "source_fingerprint": {"code": config["source_hashes"], "runtime": runtime, "source_checkpoint_sha256": "e" * 64,
                               "checkpoint_sha256": "e" * 64, "data_manifest_sha256": "f" * 64},
        "domain_counts": domain_counts(512), "counters": {"optimizer_updates": 512, "ce_positions": 4194304,
            "input_tokens": 8256 * 512, "documents": 64 * 512, "microbatches": 2048, "latent_pairs": 0, "kl_triples": 0},
        "data_cursor": 512, "evaluations": [*events, final], "checkpoints": [checkpoint], "storage_prefix": prefix,
        "wandb": {"run_url": "https://wandb.ai/test/ordinary"}}
    fusion = {"schema": report.fusion_report.SCHEMA, "status": "completed",
        "configuration": {**base, "schedule": schedule, "source_hashes": source_hashes,
            "trainable_layout": [{"name": k, "shape": [8, 8], "numel": 64} for k in report.online_report.FUSION_NAMES]},
        "source_baseline": {s: container(kind="fusion", shift=1.) for s in report.SPLITS},
        "source_baseline_small": {s: container(128, kind="fusion", shift=1., records=False) for s in report.SPLITS},
        "curves": {"mixed": fusion_curves}, "arms": {"mixed": {"metrics": {s: container(kind="fusion", shift=-.1) for s in report.SPLITS},
            "checkpoint": {"sha256": "7" * 64}, "wandb_url": "https://wandb.ai/test/fusion"}}}
    online = {"endpoints": {"source": {"state_before": initial, "checkpoint": {"sha256": "e" * 64}},
                            "mixed": {"checkpoint": {"sha256": "7" * 64}}},
        "cases": [{"endpoint": "mixed", "case": {"section": "full_context", "passes": None},
                   "metrics": {s: container(kind="online", shift=-.07) for s in report.SPLITS}}],
        "wandb": {"run_url": "https://wandb.ai/test/online"}}
    return json.loads(json.dumps([preflight, config, ordinary, fusion, online]))


def test_exact_mixed_exposure_single_ce_and_paired_references(inputs):
    before = copy.deepcopy(inputs)
    result = report.build_comparison(*inputs)
    assert inputs == before
    assert result["matched_ce_targets"] == 4194304
    assert result["trainable_parameters"] == {"ordinary": 896, "fusion_only": 128}
    for split in report.SPLITS:
        comparisons = result["comparisons"][split]
        assert comparisons["ordinary_minus_source_ordinary"]["estimate"] == pytest.approx(-.2)
        assert comparisons["ordinary_minus_fusion_K2"]["estimate"] == pytest.approx(-.1)
        assert comparisons["ordinary_minus_fusion_online"]["estimate"] == pytest.approx(-.13)
        assert comparisons["ordinary_minus_fusion_online"]["document_clusters"] == 171
    assert len(result["curves"]["ordinary"]) == 10


@pytest.mark.parametrize("fault", ["preflight", "running", "config", "plan", "budget", "lr", "objective", "beta",
    "rt", "fusion_trainable", "trainable_shape", "frozen_changed", "starting_state", "final_state", "source", "runtime",
    "data", "source_parent", "mixed_parent", "initial_score", "initial_selection", "counter", "cursor", "aux",
    "domain", "microbatches", "full_missing", "full_update", "curve_missing", "curve_duplicate", "curve_exposure",
    "ordinary_mode", "extra_pass", "target_selection", "retention", "wandb"])
def test_rejects_changed_training_contract_or_unmatched_references(inputs, fault):
    preflight, config, ordinary, fusion, online = inputs
    if fault == "preflight": preflight["status"] = "failed"
    elif fault == "running": ordinary["status"] = "running"
    elif fault == "config": config["schema"] = "other"
    elif fault == "plan": config["schedule"]["arms"]["mixed"]["batch_row_prefix"][10] += 1
    elif fault == "budget": config["schedule"]["total_ce"] -= 1
    elif fault == "lr": config["lr"] = 1e-4
    elif fault == "objective": config["objective"] = "double_ce"
    elif fault == "beta": config["beta"] = 1.
    elif fault == "rt": config["rt_layers"] = [0]
    elif fault == "fusion_trainable": config["trainable_layout"].append({"name": "backbone.fusion.state_proj.weight", "shape": [8, 8], "numel": 64})
    elif fault == "trainable_shape":
        for obj in (config, ordinary, preflight): obj["trainable_layout"][0]["numel"] = 1
    elif fault == "frozen_changed": ordinary["frozen_state_final"]["backbone.fusion.output_scale"] = "b" * 64
    elif fault == "starting_state": ordinary["full_state_initial"]["backbone.backbone.transformer.wte.weight"] = "b" * 64
    elif fault == "final_state": del ordinary["full_state_final"]["backbone.backbone.transformer.wte.weight"]
    elif fault == "source": ordinary["source_fingerprint"]["code"]["cdrm/pretrained/olmo_fbt.py"] = "c" * 64
    elif fault == "runtime": ordinary["source_fingerprint"]["runtime"] = {"device": "other"}
    elif fault == "data": config["data_manifest_sha256"] = "a" * 64
    elif fault == "source_parent": online["endpoints"]["source"]["checkpoint"]["sha256"] = "a" * 64
    elif fault == "mixed_parent": online["endpoints"]["mixed"]["checkpoint"]["sha256"] = "a" * 64
    elif fault == "initial_score": preflight["baseline"]["dev"] = container(shift=.01)
    elif fault == "initial_selection": preflight["baseline"]["dev"]["passes"][0]["document_records"][0]["document_id"] = 999
    elif fault == "counter": ordinary["counters"]["ce_positions"] -= 1
    elif fault == "cursor": ordinary["data_cursor"] = 511
    elif fault == "aux": ordinary["counters"]["kl_triples"] = 1
    elif fault == "domain": ordinary["domain_counts"]["general"]["ce_positions"] -= 1
    elif fault == "microbatches": ordinary["counters"]["microbatches"] += 1
    elif fault == "full_missing": ordinary["evaluations"].pop()
    elif fault == "full_update": ordinary["evaluations"][-1]["update"] = 511
    elif fault == "curve_missing": ordinary["evaluations"].pop(1)
    elif fault == "curve_duplicate": ordinary["evaluations"].append(copy.deepcopy(ordinary["evaluations"][0]))
    elif fault == "curve_exposure": ordinary["evaluations"][1]["input_tokens"] += 1
    elif fault == "ordinary_mode": ordinary["evaluations"][-1]["metrics"]["dev"]["passes"][0]["mode"]["enabled"] = True
    elif fault == "extra_pass": ordinary["evaluations"][-1]["metrics"]["dev"]["passes"].append(copy.deepcopy(ordinary["evaluations"][-1]["metrics"]["dev"]["passes"][0]))
    elif fault == "target_selection": ordinary["evaluations"][-1]["metrics"]["dev"]["passes"][0]["document_records"][0]["document_id"] = 999
    elif fault == "retention": ordinary["checkpoints"][0]["storage"]["sha256"] = "a" * 64
    elif fault == "wandb": ordinary["wandb"] = {}
    with pytest.raises(ValueError): report.build_comparison(*inputs)


def test_figures_and_text_keep_online_endpoints_separate_from_curves(inputs, tmp_path):
    result = report.build_comparison(*inputs)
    report.write_figures(result, tmp_path)
    assert all((tmp_path / name).stat().st_size > 1000 for name in report.FIGURES)
    text = report.markdown(result)
    assert "not a pure FBT ablation" in text
    assert "Trainable feedback CE plus constant ordinary CE" in text
    assert "O5c fusion, exact online" in text
    assert set(result["curves"]) == {"ordinary", "fusion_K2"}
    assert all(len(event["metrics"]["dev"]["passes"]) == 1 for event in result["curves"]["ordinary"])
