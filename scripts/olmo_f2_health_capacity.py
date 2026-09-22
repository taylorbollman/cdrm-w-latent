#!/usr/bin/env python3
"""Bounded F2 scale attribution and physical-batch capacity on native OLMo."""
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
from cdrm.pretrained.olmo_artifacts import (load_native_state_dict,
    load_native_tokenizer, validate_prepared_manifest)
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, optimizer_step
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_validation import require_container_gpu
from scripts.olmo_f1_common import (IntegrationCase, build_model, build_optimizer,
    changed_fixture, active_names, inference_names, boundary_digests, state_health)
from scripts.olmo_f1_observe import parameter_accounting
from scripts.olmo_f1_integrate import SOURCE_FILES as F1_SOURCES

SOURCE_FILES = tuple(sorted(set(F1_SOURCES) | {
    "scripts/olmo_f2_health_capacity.py", "scripts/olmo_f2_observe.py"}))
TERMS = ("ce", "latent", "kl")


def case_for(name, *, batch=2, length=32):
    if name not in ("ordinary", "rt", "fbt", "nextlat", "combined", "combined-k3"):
        raise ValueError("Unknown F2 case")
    return IntegrationCase(name, fbt=name in ("fbt", "combined", "combined-k3"),
        nextlat=name in ("nextlat", "combined", "combined-k3"),
        rt_layers=(0,) if name in ("rt", "combined", "combined-k3") else (),
        passes=3 if name == "combined-k3" else 2, batch_size=batch, length=length)


def group_for(name):
    return "predictor" if name.startswith("predictor.") else (
        "fusion" if name.startswith("backbone.fusion.") else "native")


def component_objectives(result):
    """Keep cross-pass derivatives attached and each term's own denominator."""
    return [(p, term, coefficient * result.weights[term] * loss.sums[term] / result.counts[term])
        for p, (coefficient, loss) in enumerate(zip(result.pass_coefficients, result.pass_losses))
        for term in TERMS if coefficient and result.weights[term] and result.counts[term]]


@torch.no_grad()
def gradient_summary(names, values, reference=None):
    """Deduplicated named parameters; stream reductions without concatenation."""
    device = next(value.device for value in values if value is not None)
    groups = {key: torch.zeros(5, dtype=torch.float64, device=device)
              for key in ("native", "fusion", "predictor", "all")}
    counts = {key: 0 for key in groups}
    for index, (name, value) in enumerate(zip(names, values)):
        if value is None:
            continue
        value = value.detach()
        # FP32 products; FP64 reductions/scalars. No full FP64 gradient copies.
        squared = (value * value).sum(dtype=torch.float64)
        finite = torch.isfinite(value).all().to(torch.float64)
        peak = value.abs().max().to(torch.float64)
        ref = None if reference is None else reference[index]
        dot = torch.zeros((), device=device, dtype=torch.float64) if ref is None else (value * ref).sum(dtype=torch.float64)
        ref_squared = torch.zeros_like(dot) if ref is None else (ref * ref).sum(dtype=torch.float64)
        for group in (group_for(name), "all"):
            groups[group][0] += squared
            groups[group][1] += dot
            groups[group][2] += ref_squared
            groups[group][3] = torch.maximum(groups[group][3], peak)
            groups[group][4] += 1-finite
            counts[group] += 1
    report = {}
    for group, scalars in groups.items():
        squared, dot, ref_squared, peak, bad = scalars.cpu().tolist()
        norm = math.sqrt(squared)
        report[group] = {"norm": norm, "max_abs": peak,
                         "finite": bad == 0 and math.isfinite(norm), "participating_tensors": counts[group]}
        if reference is not None:
            # This cosine uses the reference restricted to participating tensors.
            report[group].update(dot_with_total=dot,
                cosine_on_participating_tensors=dot/math.sqrt(squared*ref_squared)
                    if squared*ref_squared else None)
    return report


def gradient_attribution(result, named_parameters):
    names, parameters = zip(*named_parameters)
    full = torch.autograd.grad(result.total, parameters, retain_graph=True, allow_unused=True)
    aggregate = [torch.zeros_like(g) if g is not None else None for g in full]
    overall = gradient_summary(names, full)
    rows = []
    components = component_objectives(result)
    for index, (p, term, objective) in enumerate(components):
        values = torch.autograd.grad(objective, parameters,
                    retain_graph=index+1 < len(components), allow_unused=True)
        summary = gradient_summary(names, values, full)
        for group, metrics in summary.items():
            denominator = metrics["norm"] * overall[group]["norm"]
            metrics["cosine_with_full_group_gradient"] = (
                metrics["dot_with_total"]/denominator if denominator else None)
        for destination, value in zip(aggregate, values):
            if value is not None:
                if destination is None:
                    raise AssertionError("Component gradient missing from full objective")
                destination.add_(value)
        rows.append({"pass": p, "term": term, "weighted_objective": float(objective.detach()),
                     "coefficient": result.pass_coefficients[p], "groups": summary})
        del values
    closure = [None if ref is None else got-ref for got, ref in zip(aggregate, full)]
    error = gradient_summary(names, closure)
    closure_l2 = error["all"]["norm"] / max(overall["all"]["norm"], 1e-30)
    return {"total": overall, "components": rows,
        "component_sum_error": error, "component_sum_relative_l2": closure_l2,
        "closure_policy": "BF16 repeated-VJP decomposition is descriptive; tiny FP32 closure is tested",
        "passed": all(row["groups"]["all"]["finite"] for row in rows)
            and overall["all"]["finite"] and error["all"]["finite"] and math.isfinite(closure_l2)}


def finite_observations(value):
    if isinstance(value, dict):
        return all((key != "finite" or item is True) and finite_observations(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return all(finite_observations(item) for item in value)
    return True


def scalar_losses(result):
    return {"objective": float(result.total.detach()), "counts": result.counts,
        "weights": result.weights, "pass_coefficients": list(result.pass_coefficients),
        "pass_means": [{key: float(value.detach()) for key, value in loss.means.items()}
                       for loss in result.pass_losses]}


def health_case(state, tokenizer, case):
    from scripts.olmo_f2_observe import ActivationObserver
    model = build_model(state, case)
    batch = changed_fixture(tokenizer, case, 0)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    versions = {name: p._version for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    # One actual-checkpoint neutrality check, same fixed weights and input.
    reference = reference_losses = None
    if case.name == "combined":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            unobserved = model.loss_sums(batch, backbone_kwargs={"mode": case.mode()})
        reference_losses = scalar_losses(unobserved)
        reference = torch.autograd.grad(unobserved.total, [p for _, p in named], allow_unused=True)
        del unobserved
    with ActivationObserver(model, layers=(0, 1, 7, 15), max_length=max(128, case.length)) as observer:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.loss_sums(batch, backbone_kwargs={"mode": case.mode()})
    activations = observer.report()
    activation_finite = finite_observations(activations)
    losses = scalar_losses(result)
    neutrality = None
    if reference is not None:
        observed = torch.autograd.grad(result.total, [p for _, p in named], retain_graph=True, allow_unused=True)
        exact = all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                    for a, b in zip(reference, observed))
        neutrality = {"losses_exact": losses == reference_losses, "parameter_gradients_exact": exact,
                      "passed": exact and losses == reference_losses}
        del reference, observed
    gradients = gradient_attribution(result, named)
    unchanged = (versions == {name: p._version for name, p in model.named_parameters()}
        and torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state())
        and all(p.grad is None for _, p in named))
    row = {"case": asdict(case), "losses": losses, "activations": activations,
           "gradients": gradients, "neutrality": neutrality, "weights_rng_grad_buffers_unchanged": unchanged,
           "activation_finite": activation_finite,
           "passed": activation_finite and gradients["passed"] and unchanged and (neutrality is None or neutrality["passed"])}
    del result, named, model, batch, observer
    gc.collect(); torch.cuda.empty_cache()
    return row


def checkpoint_parity(state, tokenizer):
    """Same complete BF16 update, exact optimizer/model/scheduler/counter digests."""
    case = case_for("combined", batch=1, length=32)
    rows = []
    for enabled in (False, True):
        model = build_model(state, case)
        model.backbone.backbone.ordinary_activation_checkpointing = enabled
        optimizer, scheduler = build_optimizer(model)
        counters = TrainingCounters()
        metrics = optimizer_step(model, optimizer, [changed_fixture(tokenizer, case, 0)],
            config=LMTrainingConfig(precision="bf16_mixed"), scheduler=scheduler,
            counters=counters, backbone_kwargs={"mode": case.mode()})
        rows.append({"checkpointing": enabled, "metrics": metrics,
                     "boundary": boundary_digests(model, optimizer, scheduler, counters)})
        del model, optimizer, scheduler
        gc.collect(); torch.cuda.empty_cache()
    return {"case": asdict(case), "metrics_exact": rows[0]["metrics"] == rows[1]["metrics"],
            "boundary_exact": rows[0]["boundary"] == rows[1]["boundary"], "rows": rows,
            "passed": rows[0]["metrics"] == rows[1]["metrics"] and rows[0]["boundary"] == rows[1]["boundary"]}


def capacity_cell(state, tokenizer, case, *, checkpointing, warmup, repeats):
    model = build_model(state, case)
    model.backbone.backbone.ordinary_activation_checkpointing = checkpointing
    optimizer, scheduler = build_optimizer(model)
    counters = TrainingCounters()
    # Fixtures are generated on CPU and transferred before each timed interval.
    cpu_fixture = changed_fixture(tokenizer, case, 0, device="cpu")
    def make_batch(update):
        fields = {k: None if v is None else v.to("cuda") for k, v in vars(cpu_fixture).items()}
        ids = fields["input_ids"].clone()
        # Change token content; keep padding, EOS, valid targets and shape fixed.
        ids[:, :case.length-6] = ids[:, :case.length-6].roll(3*update, dims=1)
        fields["input_ids"] = ids
        return type(cpu_fixture)(**fields)
    def step(batch):
        return optimizer_step(model, optimizer, [batch], config=LMTrainingConfig(precision="bf16_mixed"),
            scheduler=scheduler, counters=counters, backbone_kwargs={"mode": case.mode()})
    torch.cuda.reset_peak_memory_stats()
    records = []
    for update in range(warmup):
        records.append(step(make_batch(update)))
    torch.cuda.synchronize()
    startup_peak = torch.cuda.max_memory_allocated()/2**30
    torch.cuda.reset_peak_memory_stats()
    times, device_times = [], []
    for update in range(warmup, warmup+repeats):
        batch = make_batch(update)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        began = time.perf_counter(); start.record()
        metrics = step(batch)
        end.record(); end.synchronize()
        times.append(time.perf_counter()-began)
        device_times.append(start.elapsed_time(end)/1000)
        records.append(metrics)
    seconds = statistics.median(times)
    peak = max(startup_peak, torch.cuda.max_memory_allocated()/2**30)
    reserved = torch.cuda.max_memory_reserved()/2**30
    final_health = state_health(model, optimizer)
    parameters = parameter_accounting(model, optimizer=optimizer,
        active_parameter_names=None, inference_parameter_names=inference_names(model, case))
    parameters["expected_active_names"] = sorted(active_names(model, case.mode()))
    parameters["activity_evidence"] = "Unobserved during capacity timing; expected names recorded separately"
    return {"case": asdict(case), "checkpointing": checkpointing, "status": "measured",
        "warmup_updates": warmup, "timed_updates": repeats, "physical_batch": case.batch_size,
        "gradient_accumulation": 1, "wall_seconds": times, "cuda_event_seconds": device_times,
        "median_wall_seconds": seconds, "valid_input_tokens_per_second": int(batch.valid_mask.sum())/seconds,
        "ce_targets_per_second": records[-1]["counts"]["ce"]/seconds,
        "valid_input_tokens_per_update": int(batch.valid_mask.sum()),
        "peak_allocated_gib": peak, "peak_reserved_gib": reserved,
        "records": records, "parameters": parameters, "post_timing_state_health": final_health,
        "scope": "Full forward/loss/backward/clip/AdamW/scheduler; excludes fixture preparation, reporting and W&B",
        "cuda_graphs": False, "passed": final_health["passed"] and all(
            math.isfinite(r["objective"]) and math.isfinite(r["gradient_norm_before_clip"]) for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("health", "capacity", "checkpoint"), required=True)
    parser.add_argument("--cases", default="ordinary,rt,fbt,nextlat,combined,combined-k3")
    parser.add_argument("--batches", default="1,8,16,32,64,128,256,512")
    parser.add_argument("--checkpointing", choices=("off", "on", "both"), default="both")
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--comfortable-gib", type=float, default=65)
    args = parser.parse_args()
    if args.warmup < 2 or args.repeats < 2 or not 20 <= args.comfortable_gib <= 70:
        parser.error("Require >=2 warm/timed updates and a20–70GiB comfortable bound")
    names = args.cases.split(",")
    for name in names:
        case_for(name)
    batches = [int(x) for x in args.batches.split(",")]
    if any(b < 1 or b > 512 for b in batches) or batches != sorted(set(batches)):
        parser.error("Physical batches must be distinct increasing integers1–512")
    runtime = require_container_gpu()
    torch.set_num_threads(4); torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {"schema": "olmo-f2-health-capacity-v1", "status": "running", "config": config,
        "runtime": {k: str(v) for k, v in runtime.items()}, "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {p: sha256_file(ROOT/p) for p in SOURCE_FILES}, "rows": [],
        "protocol_sha256": sha256_file(ROOT/"docs/reports/olmo1b-f2/protocol.md"),
        "precision": "BF16 autocast; FP32 parameters/gradients/Adam moments; TF32off",
        "qk_normalization": False, "cuda_graphs": False, "compile": False,
        "rt_layers": [0], "rt_attention": "eager exact dyadic tiling/custom reconstruction VJP"}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-step60000-f2-health-capacity", name="olmo-1b-f2-"+args.output_dir.name)
    started = time.monotonic()
    def publish(row):
        report["rows"].append(row)
        report["wandb"] = tracker.record
        write_json(args.output_dir/"report.json", report)
        index = len(report["rows"])
        if "median_wall_seconds" in row:
            prefix = row["case"]["name"]+("-checkpoint" if row["checkpointing"] else "-plain")
            metrics = {f"capacity/{prefix}/{k}": row[k] for k in
                ("physical_batch", "median_wall_seconds", "valid_input_tokens_per_second", "peak_allocated_gib")}
        elif "gradients" in row:
            prefix = row["case"]["name"]
            metrics = {f"health/{prefix}/norm": row["gradients"]["total"]["all"]["norm"],
                       f"health/{prefix}/closure_relative_l2": row["gradients"]["component_sum_relative_l2"]}
            for component in row["gradients"]["components"]:
                metrics[f"health/{prefix}/pass{component['pass']}/{component['term']}/norm"] = component["groups"]["all"]["norm"]
        else:
            metrics = {"diagnostic/passed": row.get("passed", True)}
        tracker.log(metrics, step=index)
        print({"completed": row.get("case", {}).get("name", args.stage),
            "batch": row.get("physical_batch"), "checkpointing": row.get("checkpointing"),
            "status": row.get("status", "passed" if row.get("passed") else "failed"),
            **{k: row[k] for k in ("median_wall_seconds", "peak_allocated_gib", "valid_input_tokens_per_second") if k in row}}, flush=True)
    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"config": config, "checkpoint": manifest["checkpoint"], "rt_layers": [0]})
        report["wandb"] = tracker.record
        write_json(args.output_dir/"report.json", report)
        print({"wandb": tracker.record["run_url"]}, flush=True)
        state, tokenizer = load_native_state_dict(args.artifacts), load_native_tokenizer(args.artifacts)
        if args.stage == "health":
            for name in names:
                print({"starting_health": name}, flush=True)
                row = health_case(state, tokenizer, case_for(name, length=args.length or 32))
                publish(row)
                if not row["passed"]:
                    raise AssertionError("Health/observer neutrality check failed")
        elif args.stage == "checkpoint":
            row = checkpoint_parity(state, tokenizer); publish(row)
            if not row["passed"]:
                raise AssertionError("Ordinary checkpoint complete-update parity failed")
        else:
            flags = (False, True) if args.checkpointing == "both" else (args.checkpointing == "on",)
            for name in names:
                for enabled in flags:
                    for batch in batches:
                        case = case_for(name, batch=batch, length=args.length or 512)
                        print({"starting_capacity": name, "batch": batch, "checkpointing": enabled}, flush=True)
                        try:
                            row = capacity_cell(state, tokenizer, case, checkpointing=enabled,
                                                warmup=args.warmup, repeats=args.repeats)
                        except torch.cuda.OutOfMemoryError:
                            row = {"case": asdict(case), "physical_batch": batch,
                                   "checkpointing": enabled, "status": "oom_capacity_limit", "passed": True}
                        gc.collect(); torch.cuda.empty_cache()
                        if row["status"] == "measured":
                            row["within_comfortable_memory"] = row["peak_allocated_gib"] <= args.comfortable_gib
                        publish(row)
                        if not row["passed"]:
                            raise AssertionError("Nonfinite complete capacity update")
                        if row["status"] != "measured" or not row["within_comfortable_memory"]:
                            break
        if report["source_hashes"] != {p: sha256_file(ROOT/p) for p in SOURCE_FILES}:
            raise AssertionError("F2 runtime source changed during execution")
        report["status"] = "passed"
        tracker.summary({"diagnostic/status": "passed", "diagnostic/rows": len(report["rows"])})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir/"report.json", report)


if __name__ == "__main__":
    main()
