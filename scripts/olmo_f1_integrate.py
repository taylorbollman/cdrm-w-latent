#!/usr/bin/env python3
"""F1 actual-checkpoint integration and early profiling; no quality comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo_artifacts import (
    load_native_state_dict, load_native_tokenizer, validate_prepared_manifest)
from cdrm.pretrained.olmo_reference import verify_olmo_reference_sources
from cdrm.pretrained.olmo_fbt import FBTOnlineMode
from cdrm.pretrained.lm_training import (
    LMTrainingConfig, TrainingCounters, optimizer_step,
    save_training_checkpoint, load_training_checkpoint)
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_fbt_validate import SOURCE_FILES as PRIOR_SOURCES
from scripts.olmo_validation import require_container_gpu, comparison
from scripts.olmo_lm_common import state_digests, fixture_record, verify_nextlat_sources
from scripts.olmo_f1_common import (
    IntegrationCase, build_model, build_optimizer, changed_fixture, split_rows,
    active_names, inference_names, GradientObserver, state_health,
    boundary_digests, rng_probe, state_change)
from scripts.olmo_f1_observe import parameter_accounting, summarize_profiler

PROTOCOL = "docs/reports/olmo1b-f1/protocol.md"
SOURCE_FILES = tuple(sorted({p for p in PRIOR_SOURCES if not p.startswith("docs/")} | {
    "scripts/olmo_f1_common.py", "scripts/olmo_f1_integrate.py",
    "scripts/olmo_f1_observe.py",
}))


def load_configuration(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"schema", "seed", "learning_rate", "vocab_chunk_size",
                       "profile_warmup", "profile_repeats", "cases"}:
        raise ValueError("Unexpected F1 configuration fields")
    if config["schema"] != "olmo-f1-configuration-v1":
        raise ValueError("Unexpected F1 configuration schema")
    for name in ("seed", "vocab_chunk_size", "profile_warmup", "profile_repeats"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["learning_rate"]) not in (int, float) or not 0 < config["learning_rate"] <= 1e-4:
        raise ValueError("F1 requires a bounded positive learning rate")
    cases = [IntegrationCase(**row) for row in config["cases"]]
    if not cases or len({case.name for case in cases}) != len(cases):
        raise ValueError("Case names must be nonempty and unique")
    return config, cases


def verify_fbt_reference():
    directory = ROOT / "cdrm/pretrained/_fbt_reference"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["revision"] != "7037c60924870aca6e30fac95212b0c7caee052d":
        raise ValueError("Unexpected FBT reference revision")
    for row in manifest["files"]:
        path = directory / row["file"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("FBT reference bytes changed")
    return manifest


def checkpoint_configuration(model, case, configuration):
    return {"case": asdict(case), "nextlat": model.config.to_dict(),
            "nextlat_enabled": model.enabled, "fusion": model.backbone.fusion_config.to_dict(),
            "gamma": model.gamma, "native": model.backbone.backbone.config.to_dict(),
            "training": asdict(LMTrainingConfig(precision="bf16_mixed")),
            "learning_rate": configuration["learning_rate"],
            "attention_backend": model.backbone.backbone.attention_backend,
            "attention_precision": model.backbone.backbone.attention_precision}


def observed_step(model, optimizer, scheduler, counters, batches, mode):
    observer = GradientObserver(model)
    try:
        metrics = optimizer_step(model, optimizer, batches,
                    config=LMTrainingConfig(precision="bf16_mixed"),
                    scheduler=scheduler, counters=counters, backbone_kwargs={"mode": mode})
    finally:
        observer.close()
    gradients = observer.report(active_names(model, mode))
    gradients["nonzero_accumulated_gradient"] = metrics["gradient_norm_before_clip"] > 0
    gradients["passed"] &= gradients["nonzero_accumulated_gradient"]
    return metrics, gradients


@torch.no_grad()
def final_losses(model, batch, mode):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.loss_sums(batch, backbone_kwargs={"mode": mode})
    return {"pass_coefficients": list(result.pass_coefficients), "counts": result.counts,
            "objective_weights": result.weights,
            "per_pass_means": [{key: float(value) for key, value in loss.means.items()}
                              for loss in result.pass_losses],
            "aggregate_means": {key: float(value) for key, value in result.means.items()}}


@torch.no_grad()
def cache_smoke(model, tokenizer, case):
    """Same-semantics BF16 online/cache comparison, not K2 versus online."""
    batch = changed_fixture(tokenizer, replace(case, batch_size=1, length=8), 0)
    mode = FBTOnlineMode(beta=case.beta, rt_mode=case.mode().rt_mode)
    core = model.backbone
    with torch.autocast("cuda", dtype=torch.bfloat16):
        whole = core.forward_online(batch.input_ids, document_ids=batch.document_ids,
                    attention_mask=batch.valid_mask, mode=mode, return_logits=False)
        prefix = core.forward_online(batch.input_ids[:, :3],
                    document_ids=batch.document_ids[:, :3], attention_mask=batch.valid_mask[:, :3],
                    mode=mode, return_logits=False, use_cache=True)
        suffix = core.forward_online(batch.input_ids[:, 3:],
                    document_ids=batch.document_ids, attention_mask=batch.valid_mask,
                    mode=mode, return_logits=False, past_key_values=prefix.past_key_values)
        joined = torch.cat((prefix.last_hidden_state, suffix.last_hidden_state), dim=1)
        row = comparison(joined, whole.last_hidden_state, atol=0, rtol=0)
        rejected = False
        try:
            core.forward_online(batch.input_ids[:, 3:], document_ids=batch.document_ids,
                attention_mask=batch.valid_mask, mode=replace(mode, beta=.123),
                return_logits=False, past_key_values=prefix.past_key_values)
        except ValueError:
            rejected = True
    return {"online_chunk_comparison": row, "incompatible_mode_rejected": rejected,
            "passed": row["passed"] and rejected,
            "scope": "B1/T8 exact online whole-versus-split, same BF16 execution"}


def early_profile(model, optimizer, scheduler, counters, tokenizer, case, configuration):
    """No hooks/digests inside timing; nonzero LR and changed inputs/weights."""
    update = case.updates
    training = LMTrainingConfig(precision="bf16_mixed")
    def step(batch):
        return optimizer_step(model, optimizer, [batch], config=training,
                 scheduler=scheduler, counters=counters, backbone_kwargs={"mode": case.mode()})
    for _ in range(configuration["profile_warmup"]):
        step(changed_fixture(tokenizer, case, update))
        update += 1
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, device_seconds, records = [], [], []
    for _ in range(configuration["profile_repeats"]):
        batch = changed_fixture(tokenizer, case, update)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        began = time.perf_counter()
        start.record()
        metrics = step(batch)
        end.record()
        end.synchronize()
        wall.append(time.perf_counter()-began)
        device_seconds.append(start.elapsed_time(end)/1000)
        records.append(metrics)
        update += 1
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    # The profiler is a separate complete update, excluded from throughput.
    batch = changed_fixture(tokenizer, case, update)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                  profile_memory=False, with_stack=False) as prof:
        step(batch)
        torch.cuda.synchronize()
    summary = summarize_profiler(prof, rt_selected_layers=case.rt_layers)
    seconds = statistics.median(wall)
    tokens = int(batch.valid_mask.sum())
    return {"warmup_updates": configuration["profile_warmup"],
            "timed_updates": configuration["profile_repeats"], "profiler_updates": 1,
            "wall_seconds": wall, "cuda_event_seconds": device_seconds,
            "median_wall_seconds": seconds, "median_cuda_seconds": statistics.median(device_seconds),
            "valid_input_tokens_per_second": tokens/seconds,
            "ce_targets_per_second": records[-1]["counts"]["ce"]/seconds,
            "physical_batch": case.batch_size, "logical_batch": case.batch_size,
            "input_tokens_per_update": tokens, "counts_per_update": records[-1]["counts"],
            "effective_passes": case.passes if case.fbt else 1,
            "baseline_allocated_gib": baseline/2**30, "peak_allocated_gib": peak/2**30,
            "peak_reserved_gib": reserved/2**30, "timed_steps": records, "operators": summary,
            "scope": "directional complete updates after warmup; excludes hooks, profiler, fixture creation, hashes, checkpoint and W&B I/O",
            "capacity_optimized": False}


def run_case(state, tokenizer, case, configuration, fingerprint, output_dir, on_update):
    began = time.perf_counter()
    model = build_model(state, case, chunk_size=configuration["vocab_chunk_size"])
    optimizer, scheduler = build_optimizer(model, configuration["learning_rate"])
    counters = TrainingCounters()
    before = state_digests(model)
    records, observed, active_union, inputs = [], set(), set(), []
    boundary = boundary_record = expected_boundary = None
    config = checkpoint_configuration(model, case, configuration)
    for update in range(case.updates):
        mode = case.mode(update)
        batch = changed_fixture(tokenizer, case, update)
        inputs.append(fixture_record(batch))
        # The middle update separately accumulates rows with unequal valid counts.
        batches = split_rows(batch) if update == 1 else [batch]
        metrics, gradients = observed_step(model, optimizer, scheduler, counters, batches, mode)
        observed.update(gradients["tensors"])
        active_union.update(active_names(model, mode))
        row = {"update": update+1, "mode": asdict(mode), "metrics": metrics, "gradients": gradients}
        records.append(row)
        on_update(case.name, row)
        write_json(output_dir/(case.name+"-progress.json"),
                   {"name": case.name, "updates": records, "fixtures": inputs})
        if not gradients["passed"]:
            raise AssertionError("Gradient participation/finite check failed; inspect case progress JSON")
        if case.resume and update == 1:
            boundary = output_dir / case.name / "resume-boundary.pt"
            expected_boundary = boundary_digests(model, optimizer, scheduler, counters)
            boundary_record = save_training_checkpoint(boundary, model, optimizer,
                scheduler=scheduler, counters=counters, data_cursor={"next_fixture_update": 2},
                configuration=config, source_fingerprint=fingerprint)
    after = state_digests(model)
    change = state_change(before, after, active_union)
    health = state_health(model, optimizer)
    parameters = parameter_accounting(model, optimizer=optimizer,
                      active_parameter_names=observed, inference_parameter_names=inference_names(model, case))
    result = {"name": case.name, "configuration": asdict(case), "updates": records,
              "fixtures": inputs, "state_change": change, "state_health": health,
              "parameters": parameters, "passed": change["passed"] and health["passed"]}
    if case.resume:
        expected = boundary_digests(model, optimizer, scheduler, counters)
        expected_rng = rng_probe("cuda")
        del model, optimizer, scheduler
        gc.collect(); torch.cuda.empty_cache()
        model = build_model(state, case, chunk_size=configuration["vocab_chunk_size"])
        optimizer, scheduler = build_optimizer(model, configuration["learning_rate"])
        restored = load_training_checkpoint(boundary, model, optimizer, scheduler=scheduler,
            configuration=config, source_fingerprint=fingerprint, expected_sha256=boundary_record["sha256"])
        counters = restored["counters"]
        boundary_equal = expected_boundary == boundary_digests(model, optimizer, scheduler, counters)
        cursor_equal = restored["data_cursor"] == {"next_fixture_update": 2}
        batch = changed_fixture(tokenizer, case, restored["data_cursor"]["next_fixture_update"])
        fixture_equal = fixture_record(batch) == inputs[2]
        metrics, gradients = observed_step(model, optimizer, scheduler, counters, [batch], case.mode(2))
        actual = boundary_digests(model, optimizer, scheduler, counters)
        continuation_equal = expected == actual
        rng_equal = expected_rng == rng_probe("cuda")
        metrics_equal = metrics == records[2]["metrics"]
        resume = {"checkpoint": boundary_record, "loaded_boundary_exact": boundary_equal,
                  "cursor_exact": cursor_equal, "next_fixture_exact": fixture_equal,
                  "next_update_state_exact": continuation_equal, "next_rng_exact": rng_equal,
                  "next_update_metrics_exact": metrics_equal, "gradients": gradients,
                  "passed": all((boundary_equal, cursor_equal, fixture_equal,
                                 continuation_equal, rng_equal, metrics_equal, gradients["passed"]))}
        if resume["passed"]:
            boundary.unlink()
        resume["disposable_checkpoint_deleted"] = not boundary.exists()
        result["resume"] = resume
        result["passed"] &= resume["passed"]
    if case.name == "rt-fbt-nextlat":
        result["cache"] = cache_smoke(model, tokenizer, case)
        result["passed"] &= result["cache"]["passed"]
    if case.profile and result["passed"]:
        result["profile"] = early_profile(model, optimizer, scheduler, counters,
                                         tokenizer, case, configuration)
        result["post_profile_health"] = state_health(model, optimizer)
        result["passed"] &= result["post_profile_health"]["passed"]
    batch = changed_fixture(tokenizer, case, max(0, counters.optimizer_updates-1))
    result["endpoint_losses"] = final_losses(model, batch, case.mode(case.updates-1))
    result["endpoint_losses_finite"] = all(math.isfinite(value)
        for losses in [*result["endpoint_losses"]["per_pass_means"],
                       result["endpoint_losses"]["aggregate_means"]] for value in losses.values())
    result["passed"] &= result["endpoint_losses_finite"]
    result["counters"] = asdict(counters)
    result["elapsed_seconds_including_checks"] = time.perf_counter()-began
    del model, optimizer, scheduler
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, default=ROOT/"configs/olmo_f1_integration.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", help="Comma-separated subset for a bounded preflight; recorded explicitly")
    args = parser.parse_args()
    runtime = require_container_gpu()
    configuration, cases = load_configuration(args.configuration)
    if args.cases:
        selected = args.cases.split(",")
        if len(set(selected)) != len(selected) or set(selected)-{case.name for case in cases}:
            raise ValueError("Unknown or duplicated selected case")
        cases = [case for case in cases if case.name in selected]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"configuration.json", configuration)
    torch.set_num_threads(4)
    torch.manual_seed(configuration["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-f1-integration-v1", "status": "running",
              "runtime": {k: str(v) for k, v in runtime.items()},
              "config": configuration, "config_sha256": sha256_file(args.configuration),
              "protocol_sha256": sha256_file(ROOT/PROTOCOL),
              "source_hashes": {p: sha256_file(ROOT/p) for p in SOURCE_FILES},
              "requested_cases": [case.name for case in cases], "cases": [],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "F1 bounded actual-checkpoint integration and early directional profiling, no quality comparison",
              "precision": "bf16_mixed", "fp32_parameters_gradients_moments": True,
              "compile": False, "cuda_graphs": False, "distributed": False,
              "qk_normalization": False, "rt_attention": "native eager dyadic tiling, explicit custom VJP"}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                    group="olmo1b-step60000-f1-integration", name="olmo-1b-f1-"+args.output_dir.name)
    started, log_step = time.monotonic(), 0
    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report.update(checkpoint=manifest["checkpoint"],
                      native_reference=verify_olmo_reference_sources(),
                      nextlat_reference=verify_nextlat_sources(), fbt_reference=verify_fbt_reference())
        tracker.start({"scope": report["scope"], "checkpoint_sha256": report["checkpoint"]["sha256"],
                       "configuration": configuration, "requested_cases": report["requested_cases"]})
        report["wandb"] = tracker.record
        write_json(args.output_dir/"report.json", report)
        print({"wandb": tracker.record["run_url"], "cases": report["requested_cases"]}, flush=True)
        state, tokenizer = load_native_state_dict(args.artifacts), load_native_tokenizer(args.artifacts)
        fingerprint = {"checkpoint_sha256": report["checkpoint"]["sha256"],
                       "source_hashes": report["source_hashes"], "config_sha256": report["config_sha256"]}
        def on_update(name, row):
            nonlocal log_step
            log_step += 1
            metrics = row["metrics"]
            tracker.log({f"cases/{name}/objective": metrics["objective"],
                         f"cases/{name}/update": row["update"],
                         f"cases/{name}/gradient_check": row["gradients"]["passed"],
                         **{f"cases/{name}/{key}": value for key, value in metrics["loss_means"].items()}},
                         step=log_step)
            print({"case": name, "update": row["update"], "objective": metrics["objective"],
                   "gradients_passed": row["gradients"]["passed"]}, flush=True)
        for case in cases:
            report["current_case"] = case.name
            write_json(args.output_dir/"report.json", report)
            print({"case_start": case.name}, flush=True)
            row = run_case(state, tokenizer, case, configuration, fingerprint, args.output_dir, on_update)
            report["cases"].append(row)
            write_json(args.output_dir/(case.name+".json"), row)
            write_json(args.output_dir/"report.json", report)
            log_step += 1
            metrics = {f"cases/{case.name}/passed": row["passed"]}
            if "profile" in row:
                metrics.update({f"profiles/{case.name}/{key}": row["profile"][key] for key in
                    ("median_wall_seconds", "valid_input_tokens_per_second", "peak_allocated_gib")})
            tracker.log(metrics, step=log_step)
            print({"case_complete": case.name, "passed": row["passed"],
                   "seconds": row["elapsed_seconds_including_checks"]}, flush=True)
            if not row["passed"]:
                raise AssertionError("F1 case failed; retain evidence and localize before continuing")
        if report["source_hashes"] != {p: sha256_file(ROOT/p) for p in SOURCE_FILES}:
            raise AssertionError("Runtime source changed during execution")
        report["status"] = "passed"
        tracker.summary({"integration/status": "passed", "integration/cases": len(cases)})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__,
                      failed_case=report.get("current_case"))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir/"report.json", report)
    print({"status": report["status"], "report": str(args.output_dir/"report.json")}, flush=True)


if __name__ == "__main__":
    main()
