"""F3 reports preserve numerical checks, timing scope and exact source lineage."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import olmo_f3_report as report


def write(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)
    return path


def health():
    return {"passed": True, "nonfinite_parameters": [], "nonfinite_optimizer_tensors": []}


def metrics(configuration, update):
    b, t = configuration["batch_size"], configuration["length"]
    enabled = configuration["case"] != "rt"
    counts = {"ce": b*t//2, "latent": b*(t-1) if enabled else 0, "kl": b*t//2 if enabled else 0}
    weights = {"ce": 1., "latent": float(enabled), "kl": float(enabled)}
    return {"schema": "olmo-lm-optimizer-step-v1", "update_completed": True,
        "counts": counts, "objective_weights": weights, "loss_sums": {k: float(v) for k, v in counts.items()},
        "loss_means": weights, "objective": sum(weights.values()), "max_grad_norm": 1.,
        "gradient_norm_before_clip": 10., "lr_used": [.0001], "lr_next": [.0001],
        "counters": {"optimizer_updates": update, "microbatches": update, "documents": update*b,
            "input_tokens": update*b*t, "ce_positions": update*counts["ce"],
            "latent_pairs": update*counts["latent"], "kl_triples": update*counts["kl"]}}


def comparison():
    return {"bitwise_equal": True, "finite": True, "relative_l2": 0., "max_abs": 0.,
        "reference_max_abs": 2., "relative_l2_limit": 1e-5, "max_abs_limit": 2.1e-5, "passed": True}


def timing(seconds):
    return {"wall_seconds": [seconds]*3, "cuda_seconds": [seconds*.9]*3,
        "median_wall_seconds": seconds, "median_cuda_seconds": seconds*.9}


def capacity_row(config, replay):
    records = [metrics(config, index) for index in range(4, 7)]
    peak = 40. if replay else 30.
    return {"case": config["case"], "batch_size": config["batch_size"], "length": config["length"],
        "replay": replay, "checkpointing": config["checkpointing"] == "on", "capture_seconds": 2. if replay else None,
        "warmup_updates": 3, "timed_updates": 3, "gradient_initialization_backward": 1,
        "backward_only_warmup": 10, "graph_capture_backward": int(replay),
        "input_tokens_per_update": config["batch_size"]*config["length"], "counts": records[0]["counts"],
        "full_step": {**timing(1. if replay else 2.),
            "valid_input_tokens_per_second": config["batch_size"]*config["length"]/(1. if replay else 2.)},
        "forward_loss_backward": timing(.25 if replay else 1.),
        "capture_memory": {"allocated_gib": peak-5, "reserved_gib": peak,
            "peak_allocated_gib": peak-2, "peak_reserved_gib": peak+3},
        "peak_allocated_gib": peak, "peak_reserved_gib": peak+5,
        "within_comfortable_budget": True, "records": records, "post_timing_health": health(),
        "full_step_scope": "input copies, fwd loss bwd, clip, optimizer, scheduler",
        "region_scope": "forward loss backward only", "passed": True}


@pytest.fixture
def evidence(tmp_path):
    project, runtime = tmp_path/"project", tmp_path/"runs"
    protocol = write(project, "docs/reports/olmo1b-f3/protocol.md", "frozen protocol")
    sources = {}
    for name in report.ESSENTIAL_SOURCES:
        sources[name] = report.file_digest(write(project, name, "frozen source"))["sha256"]
    configuration = {"case": "combined", "stage": "correctness", "checkpointing": "on", "batch_size": 1,
        "length": 32, "warmup": 10, "updates": 3, "repeats": 3, "comfortable_gib": 65.,
        "precision": "bf16_mixed", "ordinary_sdpa_backend": "flash", "deterministic_algorithms": True,
        "autocast_weight_cache": False, "gradient_accumulation": 1, "rt_layers": [0]}
    base = {"schema": report.REPORT_SCHEMA, "status": "passed", "stage": "complete", "finished_utc": "now",
        "configuration": configuration, "source_hashes": sources,
        "protocol_sha256": report.file_digest(protocol)["sha256"], "capture_succeeded": True,
        "checkpoint": {"sha256": report.CHECKPOINT_SHA256, "size_bytes": report.CHECKPOINT_SIZE},
        "wandb": {"status": "synced", "run_url": "https://wandb.ai/taylorbollman/test/runs/id"}, "rows": []}
    active = ["native.weight", "predictor.weight"]
    checks = [{"name": name, "passed": True, "ownership_matches": True, "all_bitwise_equal": True,
        "gradients": {name: comparison() for name in active},
        "losses": {f"pass{p}/{term}": comparison() for p in range(2) for term in report.TERMS}}
        for name in report.TENSOR_CHECKS]
    eager_metrics = [metrics(configuration, index) for index in range(1, 4)]
    boundary = {"model": {"weight": "a"*64}, "optimizer": {"moment": "b"*64},
        "scheduler": {"last_epoch": 3}, "counters": eager_metrics[-1]["counters"]}
    check = {"name": report.UPDATE_CHECK, "passed": True, "updates_per_arm": 3, "physical_optimizer_updates": 6,
        "metrics_exact": True, "model_optimizer_scheduler_counters_exact": True, "weights_changed": True,
        "arms": [{"replay": replay, "metrics": copy.deepcopy(eager_metrics),
            "boundary": copy.deepcopy(boundary), "health": health()} for replay in (False, True)]}
    correct = copy.deepcopy(base) | {"checks": checks[:3]+[check]+checks[3:], "capture_seconds": 1.,
        "parameters": {"active": {"names": active}}, "backward_preparation": {"warmup": 11, "capture": 1, "replay": 8},
        "memory": {"peak_allocated_gib": 30., "peak_reserved_gib": 40.},
        "observed_attention_operators": ["aten::_scaled_dot_product_flash_attention"]}
    capacity = copy.deepcopy(base)
    capacity["configuration"].update(stage="capacity", batch_size=32, length=512)
    capacity["rows"] = [capacity_row(capacity["configuration"], replay) for replay in (False, True)]
    capacity["checks"] = [{"name": label+"_complete_updates_finite", "passed": True} for label in ("eager", "graph")]
    directories = [runtime/"f3-correctness", runtime/"f3-capacity"]
    for directory, data in zip(directories, (correct, capacity)):
        write(directory, "report.json", data)
    return SimpleNamespace(project=project, runtime=runtime, directories=directories,
        correct=correct, capacity=capacity)


def summarize(e):
    return report.summarize(e.directories, project_root=e.project)


def test_correctness_and_capacity_have_distinct_scope_and_update_counts(evidence):
    summary = summarize(evidence)
    assert summary["status"] == "passed"
    assert summary["correctness"][0]["all_gradients_bitwise_equal"]
    assert summary["correctness"][0]["physical_optimizer_updates"] == 6
    assert summary["capacity"][0]["full_step_speedup"] == 2.
    assert summary["capacity"][0]["region_speedup"] == 4.
    assert summary["physical_optimizer_updates"] == {"prepared_eager": 12, "graph": 6, "total": 18}
    assert summary["optimizer_updates_by_comparison_arm"] == {"prepared_eager": 9, "graph": 9, "total": 18}
    markdown = report.markdown(summary)
    assert "not all-gradient/full-state parity" in summary["capacity"][0]["validation_scope"]
    assert "all-layer RT" in markdown and "Input tokens/s" in markdown and "not quality" in markdown
    assert "Setup/warmup + capture seconds" in markdown
    assert "initial gradient discovery" in summary["capture_seconds_scope"]
    assert summary["capacity"][0]["rows"][1]["capture_seconds"] == 2.
    assert "Current postcapture allocated / reserved" in markdown
    assert "35.00 / 40.00 | 38.00 / 43.00" in markdown
    assert "not the continuing graph-pool footprint" in markdown
    assert "12 prepared-eager updates + 6 graph-replay updates" in markdown
    assert "three eager preparation updates" in markdown
    assert "Current memory after setup" in summary["memory_scopes"]["capture_memory.allocated_gib/reserved_gib"]


@pytest.mark.parametrize("change", ["missing_check", "hidden_failure", "ownership", "widened_budget", "max_error",
    "bitwise", "moment", "counter", "backward_count", "timing_nan", "timing_median", "throughput", "memory"])
def test_report_cannot_override_failed_or_inconsistent_evidence(evidence, change):
    data, directory = evidence.correct, evidence.directories[0]
    if change == "missing_check": data["checks"].pop()
    elif change == "hidden_failure": data["checks"][0]["gradients"]["native.weight"]["finite"] = False
    elif change == "ownership": del data["checks"][0]["gradients"]["native.weight"]
    elif change == "widened_budget": data["checks"][0]["gradients"]["native.weight"]["relative_l2_limit"] = .01
    elif change == "max_error": data["checks"][0]["gradients"]["native.weight"]["max_abs"] = .5
    elif change == "bitwise": data["checks"][0]["all_bitwise_equal"] = False
    elif change == "moment": data["checks"][3]["arms"][1]["boundary"]["optimizer"]["moment"] = "c"*64
    elif change == "counter": data["checks"][3]["arms"][1]["metrics"][0]["counters"]["optimizer_updates"] = 5
    elif change == "backward_count": data["backward_preparation"]["replay"] = 99
    else:
        data, directory = evidence.capacity, evidence.directories[1]
        row = data["rows"][1]
        if change == "timing_nan": row["full_step"]["wall_seconds"][0] = float("nan")
        elif change == "timing_median": row["full_step"]["median_wall_seconds"] = 10.
        elif change == "throughput": row["full_step"]["valid_input_tokens_per_second"] *= 2
        elif change == "memory": row["within_comfortable_budget"] = False
    write(directory, "report.json", data)
    with pytest.raises(ValueError): summarize(evidence)


def test_failed_run_partial_metrics_never_enter_performance(evidence, tmp_path):
    data = evidence.capacity
    data.update(status="failed", stage="capacity_timing", error_type="RuntimeError", error_message="OOM after eager arm")
    data["wandb"]["status"] = "synced_failed_experiment"
    data["rows"][0]["full_step"]["valid_input_tokens_per_second"] = float("nan")
    write(evidence.directories[1], "report.json", data)
    summary = summarize(evidence)
    assert summary["status"] == "completed_with_failed_diagnostics" and not summary["capacity"]
    assert summary["failed_diagnostics"][0]["excluded_partial_capacity_rows"] == 2
    assert summary["physical_optimizer_updates"]["total"] == 6
    assert not summary["runs"][1]["used_for_performance"]
    assert "OOM after eager arm" in report.markdown(summary)
    write(tmp_path, "throughput.png", "stale plot")
    assert report.plot(summary, tmp_path) == [] and not (tmp_path/"throughput.png").exists()


def test_capture_blocker_is_preserved_and_not_promoted(evidence):
    data = evidence.correct
    data.update(status="capture_blocked", stage="capture", error_type="RuntimeError", error_message="unsupported capture",
        checks=[], capture_succeeded=False)
    data["wandb"]["status"] = "synced_failed_experiment"
    write(evidence.directories[0], "report.json", data)
    summary = summarize(evidence)
    assert not summary["correctness"] and len(summary["capacity"]) == 1
    assert summary["failed_diagnostics"][0]["status"] == "capture_blocked"


def test_memory_stop_without_graph_stays_unpaired(evidence):
    data = evidence.capacity
    data["rows"] = data["rows"][:1]; data["checks"] = data["checks"][:1]
    data.update(stopped_at_memory_budget=True, capture_succeeded=False)
    data["rows"][0].update(peak_allocated_gib=66., peak_reserved_gib=70., within_comfortable_budget=False)
    write(evidence.directories[1], "report.json", data)
    summary = summarize(evidence)
    assert not summary["capacity"][0]["paired"] and summary["capacity"][0]["full_step_speedup"] is None
    assert not summary["runs"][1]["used_for_performance"]


def test_exact_run_snapshot_preserves_old_successful_source(evidence):
    name = "cdrm/pretrained/static_training.py"
    for directory in evidence.directories:
        write(directory, "source-snapshot/"+name, (evidence.project/name).read_text())
    write(evidence.project, name, "changed after run")
    summary = summarize(evidence)
    assert not summary["runs"][0]["source_lineage"][name]["current_matches"]
    assert summary["runs"][0]["source_lineage"][name]["exact_snapshot"]
    write(evidence.directories[0], "source-snapshot/"+name, "corrupt old source")
    with pytest.raises(ValueError, match="snapshot"): summarize(evidence)


def test_unreconstructable_source_or_wrong_checkpoint_fails(evidence):
    name = "cdrm/pretrained/static_training.py"
    write(evidence.project, name, "unretained change")
    with pytest.raises(ValueError, match="Unreconstructable"): summarize(evidence)
    write(evidence.project, name, "frozen source")
    evidence.correct["checkpoint"]["sha256"] = "a"*64
    write(evidence.directories[0], "report.json", evidence.correct)
    with pytest.raises(ValueError, match="checkpoint"): summarize(evidence)


def test_capacity_plot_exports_paired_complete_updates(evidence, tmp_path):
    summary = summarize(evidence)
    paths = report.plot(summary, tmp_path)
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.stat().st_size > 1000 for path in paths)
