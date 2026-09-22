"""Completed-report validation uses synthetic metadata, never model allocation."""
import copy
from dataclasses import asdict
import json
import math

import pytest

from scripts.olmo_f1_common import IntegrationCase
from scripts import olmo_f1_report as report


def _parameters(case):
    shapes = {"backbone.backbone.transformer.wte.weight": [50304, 2048]}
    for index in range(16):
        prefix = f"backbone.backbone.transformer.blocks.{index}."
        shapes.update({prefix + "att_proj.weight": [6144, 2048], prefix + "attn_out.weight": [2048, 2048],
                       prefix + "ff_proj.weight": [16384, 2048], prefix + "ff_out.weight": [2048, 8192]})
    shapes.update({"backbone.fusion.state_proj.weight": [2048, 2048],
                   "backbone.fusion.token_gate.weight": [2048, 2048]})
    if case.nextlat:
        shapes.update({"predictor.mlp.0.weight": [6528, 4096], "predictor.mlp.2.weight": [6528, 6528],
                       "predictor.mlp.4.weight": [2048, 6528], "predictor.norm_x.weight": [4096]})
    native = [name for name in shapes if name.startswith("backbone.backbone.")]
    fusion = [name for name in shapes if name.startswith("backbone.fusion.")]
    predictor = [name for name in shapes if name.startswith("predictor.")]
    trainable = native + (fusion if case.fbt else []) + predictor
    active = native + (fusion if case.fbt and (case.beta > 0 or case.transition) else []) + predictor
    scopes = {"registered": list(shapes), "requires_grad": trainable, "active": active,
              "optimizer_owned": trainable, "inference": native + (fusion if case.fbt else [])}
    rows = [{"name": name, "aliases": [name], "shape": shape, "dtype": "torch.float32", "device": "cuda:0",
             "requires_grad": name in trainable, "parameter_count": math.prod(shape),
             "parameter_bytes": math.prod(shape) * 4} for name, shape in shapes.items()]
    return {"schema": "olmo-f1-parameter-accounting-v1", "parameter_records": rows,
        "optimizer_groups": [trainable], "resident_parameter_storage": {"known_allocated_bytes": sum(math.prod(s)*4 for s in shapes.values())},
        **{key: {"names": names, "tensor_count": len(names), "parameter_count": sum(math.prod(shapes[n]) for n in names),
                 "parameter_bytes": sum(math.prod(shapes[n])*4 for n in names)} for key, names in scopes.items()}}


def _counts(case):
    lengths = [case.length - (5 if index % 2 else 0) for index in range(case.batch_size)]
    ce = sum((n+1)//2 for n in lengths)
    return {"ce": ce, "latent": sum(n-1 for n in lengths) if case.nextlat else 0,
            "kl": ce if case.nextlat else 0}


def _counters(case, update):
    counts = _counts(case)
    return {"optimizer_updates": update,
            "microbatches": update + (case.batch_size-1 if case.updates >= 2 and update >= 2 else 0),
            "documents": case.batch_size * update,
            "input_tokens": sum(case.length - (5 if index % 2 else 0) for index in range(case.batch_size))*update,
            "ce_positions": counts["ce"]*update, "latent_pairs": counts["latent"]*update, "kl_triples": counts["kl"]*update}


def _metrics(case, update):
    counts = _counts(case)
    weights = {"ce": 1., "latent": float(case.nextlat), "kl": float(case.nextlat)}
    means = {key: 2. if counts[key] else 0. for key in counts}
    return {"schema": "olmo-lm-optimizer-step-v1", "update_completed": True, "counts": counts,
            "counters": _counters(case, update), "objective_weights": weights, "loss_means": means,
            "loss_sums": {key: means[key]*counts[key] for key in counts},
            "objective": sum(means[key]*weights[key] for key in counts),
            "gradient_norm_before_clip": 2., "max_grad_norm": 1., "lr_used": [1e-5], "lr_next": [1e-5]}


def _fixture(case, update):
    result = {key: [] for key in ("input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask")}
    for index in range(case.batch_size):
        n = case.length - (5 if index % 2 else 0)
        padding = case.length-n
        valid = [True]*n+[False]*padding
        response = [False]*(n//2)+[True]*(n-n//2)+[False]*padding
        result["input_ids"].append([10+update]*(n-1)+[50279]+[1]*padding)
        result["valid_mask"].append(valid)
        result["document_ids"].append([index]*n+[-1]*padding)
        result["ce_mask"].append(response); result["kl_mask"].append(response)
        result["latent_mask"].append(valid)
    return result


def _gradients(names):
    return {"passed": True, "nonzero_accumulated_gradient": True, "missing": [], "unexpected": [],
            "tensors": {name: {"finite": True, "backward_contributions": 1,
                               "contribution_norm_sum": 1., "max_abs": .1} for name in names}}


def _health():
    return {"passed": True, "nonfinite_parameters": [], "nonfinite_optimizer_tensors": []}


def _case(case):
    params = _parameters(case); active = params["active"]["names"]
    row = {"name": case.name, "configuration": json.loads(json.dumps(asdict(case))), "passed": True,
           "parameters": params, "fixtures": [_fixture(case, index) for index in range(case.updates)],
           "updates": [{"update": index+1, "mode": json.loads(json.dumps(asdict(case.mode(index)))),
                        "metrics": _metrics(case, index+1), "gradients": _gradients([name for name in active
                            if not (case.mode(index).beta == 0 and name.startswith("backbone.fusion."))])}
                       for index in range(case.updates)],
           "state_change": {"passed": True, "changed": list(active), "unchanged_active": [], "unexpected_changes": []},
           "state_health": _health(), "counters": _counters(case, case.updates),
           "elapsed_seconds_including_checks": 20., "endpoint_losses_finite": True}
    passes = case.passes if case.fbt else 1
    row["endpoint_losses"] = {"counts": _counts(case), "pass_coefficients": [1.]+[1./(passes-1)]*(passes-1) if passes>1 else [1.],
        "objective_weights": {"ce": 1., "latent": float(case.nextlat), "kl": float(case.nextlat)},
        "per_pass_means": [{key: 2. if value else 0. for key, value in _counts(case).items()} for _ in range(passes)],
        "aggregate_means": {key: 2. if value else 0. for key, value in _counts(case).items()}}
    if case.resume:
        row["resume"] = {key: True for key in ("passed", "loaded_boundary_exact", "cursor_exact", "next_fixture_exact",
            "next_update_state_exact", "next_rng_exact", "next_update_metrics_exact", "disposable_checkpoint_deleted")}
        row["resume"].update(gradients=_gradients(active), checkpoint={"schema": "olmo-lm-training-checkpoint-v1",
            "optimizer_updates": 2, "size_bytes": 1000, "sha256": "b"*64})
    if case.name == "rt-fbt-nextlat":
        row["cache"] = {"passed": True, "incompatible_mode_rejected": True,
            "online_chunk_comparison": {"passed": True, "finite": True, "max_abs": 0., "difference_l2": 0., "atol": 0., "rtol": 0.}}
    if case.profile:
        tokens = case.batch_size*case.length-5*(case.batch_size//2)
        op = {"name": "aten::_scaled_dot_product_cudnn_attention", "count": 16,
              "self_cpu_time_us": 1., "cpu_time_us": 2., "self_device_time_us": 3., "device_time_us": 4.}
        row["profile"] = {"warmup_updates": 3, "timed_updates": 3, "profiler_updates": 1,
            "wall_seconds": [1., 2., 3.], "cuda_event_seconds": [.5, 1., 1.5],
            "median_wall_seconds": 2., "median_cuda_seconds": 1., "valid_input_tokens_per_second": tokens/2,
            "ce_targets_per_second": _counts(case)["ce"]/2, "input_tokens_per_update": tokens,
            "counts_per_update": _counts(case), "physical_batch": case.batch_size, "logical_batch": case.batch_size,
            "effective_passes": passes, "capacity_optimized": False, "baseline_allocated_gib": 20.,
            "peak_allocated_gib": 25., "peak_reserved_gib": 30.,
            "timed_steps": [_metrics(case, case.updates+4+index) for index in range(3)],
            "operators": {"schema": "olmo-f1-profiler-summary-v1", "rt_selected_layers": list(case.rt_layers),
                "attention_dispatch": {"cudnn_sdpa": {"observed": True, "event_count": 16, "evidence": [op]}},
                "top_cpu_operators": [op], "top_device_operators": [op]}}
        row["post_profile_health"] = _health()
        row["counters"] = _counters(case, case.updates+7)
    return row


def _report(*cases, subset=None):
    cases = cases or (IntegrationCase("ordinary", resume=True),)
    selected = cases if subset is None else [case for case in cases if case.name in subset]
    return {"schema": "olmo-f1-integration-v1", "status": "passed", "scope": "bounded fixture integration",
        "started_utc": "2026-09-22T00:00:00Z", "finished_utc": "2026-09-22T00:10:00Z", "elapsed_seconds": 600.,
        "precision": "bf16_mixed", "fp32_parameters_gradients_moments": True,
        "compile": False, "cuda_graphs": False, "distributed": False, "qk_normalization": False,
        "checkpoint": {"sha256": report.CHECKPOINT_SHA256, "size_bytes": 4707065440,
                       "url": f"https://huggingface.co/allenai/OLMo-1B/resolve/{report.MODEL_REVISION}/model.safetensors"},
        "native_reference": {"revision": "native-reference"}, "nextlat_reference": {"revision": "nextlat-reference"},
        "fbt_reference": {"revision": "fbt-reference"}, "source_hashes": {"scripts/olmo_f1_common.py": "a"*64},
        "config_sha256": "c"*64, "protocol_sha256": "d"*64,
        "runtime": {"torch": "fixture", "cuda": "fixture", "gpu": "metadata only"},
        "wandb": {"status": "synced", "enabled": True, "mode": "online", "run_url": "https://wandb.ai/taylorbollman/fixture/runs/test"},
        "config": {"schema": "olmo-f1-configuration-v1", "seed": 1, "learning_rate": 1e-5, "vocab_chunk_size": 128,
                   "profile_warmup": 3, "profile_repeats": 3, "cases": [json.loads(json.dumps(asdict(case))) for case in cases]},
        "requested_cases": [case.name for case in selected], "cases": [_case(case) for case in selected]}


def test_full_configured_matrix_accounting_and_actual_scope():
    config = json.loads((report.ROOT/"configs/olmo_f1_integration.json").read_text())
    source = _report(*(IntegrationCase(**row) for row in config["cases"]))
    ledger = report.build_ledger(source)
    assert ledger["coverage"] == "full_configured_matrix" and len(ledger["cases"]) == 18
    assert ledger["update_accounting"] == {"original_observed_updates": 44, "warmup_updates": 12,
        "timed_updates": 12, "profiler_updates": 4, "resume_replay_updates": 2,
        "retained_counter_updates": 72, "physical_optimizer_executions": 74}
    text = report.markdown(ledger)
    assert "18/18" in text and "74 optimizer executions" in text and "cuDNN SDPA" in text
    assert "Flash/CuTE" in text and "not establish language-model quality" in text
    assert "82,726,912" not in text  # Table reports full scopes, not a second guessed architecture count.
    json.dumps(ledger, allow_nan=False)


def test_subset_stays_subset_and_unrequested_features_are_not_cleared():
    ordinary = IntegrationCase("ordinary", resume=True)
    combined = IntegrationCase("rt-fbt-nextlat", rt_layers=(0,), fbt=True, nextlat=True, resume=True)
    ledger = report.build_ledger(_report(ordinary, combined, subset=["ordinary"]))
    assert ledger["coverage"] == "explicit_subset"
    assert ledger["untested_configured_cases"] == ["rt-fbt-nextlat"]
    assert ledger["cases"][0]["checks"]["online_cache_B1_T8"] == "untested"
    assert "1/2" in report.markdown(ledger)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(status="running"), lambda r: r["cases"][0].update(passed=False),
    lambda r: r["cases"][0]["state_health"].update(passed=False),
    lambda r: r["cases"][0]["state_health"]["nonfinite_optimizer_tensors"].append("bad"),
    lambda r: r["cases"][0]["updates"][0]["metrics"].update(objective=float("nan")),
    lambda r: r["cases"][0]["updates"][0]["metrics"].update(objective=123),
    lambda r: r["cases"][0]["updates"][0]["gradients"].update(missing=["missing"]),
    lambda r: next(iter(r["cases"][0]["updates"][0]["gradients"]["tensors"].values())).update(finite=False),
    lambda r: r["cases"][0]["resume"].update(next_update_state_exact=False),
    lambda r: r["cases"][0]["resume"].update(next_rng_exact=False),
    lambda r: r["cases"][0]["resume"].update(disposable_checkpoint_deleted=False),
    lambda r: r["cases"][0]["counters"].update(optimizer_updates=4),
    lambda r: r["cases"][0]["parameters"]["registered"].update(parameter_count=1),
    lambda r: r["cases"][0]["parameters"]["optimizer_groups"][0].append(r["cases"][0]["parameters"]["optimizer_groups"][0][0]),
    lambda r: r["cases"][0]["state_change"]["unexpected_changes"].append("backbone.fusion.output_scale"),
    lambda r: r["cases"][0]["endpoint_losses"].update(pass_coefficients=[.5]),
    lambda r: r["checkpoint"].update(sha256="0"*64),
    lambda r: r["wandb"].update(status="running"),
    lambda r: r["requested_cases"].append("unknown"), lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
])
def test_failed_or_contradictory_nested_evidence_cannot_be_upgraded(mutate):
    source = _report(); mutate(source)
    with pytest.raises(ValueError): report.build_ledger(source)


@pytest.mark.parametrize("mutate", [
    lambda r: r["cases"][0]["profile"].update(median_wall_seconds=123),
    lambda r: r["cases"][0]["profile"].update(valid_input_tokens_per_second=123),
    lambda r: r["cases"][0]["profile"].update(peak_allocated_gib=40),
    lambda r: r["cases"][0]["profile"]["wall_seconds"].append(1),
    lambda r: r["cases"][0]["profile"]["timed_steps"][0].update(update_completed=False),
    lambda r: r["cases"][0]["profile"]["operators"]["attention_dispatch"]["cudnn_sdpa"].update(observed=False),
    lambda r: r["cases"][0]["profile"]["operators"]["top_device_operators"][0].update(self_device_time_us=float("inf")),
    lambda r: r["cases"][0]["post_profile_health"].update(passed=False),
])
def test_profile_finiteness_coverage_and_derived_results(mutate):
    source = _report(IntegrationCase("ordinary-profile", length=512, batch_size=1, updates=1, profile=True))
    mutate(source)
    with pytest.raises(ValueError): report.validate_report(source)


def test_cache_exactness_and_required_check_presence():
    source = _report(IntegrationCase("rt-fbt-nextlat", fbt=True, nextlat=True, rt_layers=(0,), resume=True))
    report.validate_report(source)
    source["cases"][0]["cache"]["online_chunk_comparison"]["max_abs"] = 1e-8
    with pytest.raises(ValueError): report.validate_report(source)
    source["cases"][0].pop("cache")
    with pytest.raises(ValueError): report.validate_report(source)


def test_cli_rejects_failed_report_before_writing_outputs(tmp_path, monkeypatch):
    source = _report(); source["status"] = "failed"
    input_file = tmp_path / "failed.json"; input_file.write_text(json.dumps(source))
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["olmo_f1_report", "--report", str(input_file), "--output-dir", str(output)])
    with pytest.raises(ValueError): report.main()
    assert not output.exists()
