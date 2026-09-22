#!/usr/bin/env python3
"""Bounded native RT backward attribution, separate from full-update timing."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cdrm.pretrained import olmo_tiled
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import TrainingCounters
from scripts.olmo_f3_graph_training import (
    configure_determinism, require_container_gpu, validate_prepared_manifest,
    load_native_state_dict, load_native_tokenizer, case_for, build_optimizer,
    changed_batch, backend_context, state_health, timed, OnlineTracker,
    loss_snapshot,
)

PROTOCOL = ROOT / "docs/reports/olmo1b-f3c/protocol.md"
HELPER_PHASES = {
    "_project": "RT/primal_project",
    "_finish": "RT/primal_finish",
    "_add_tile": "RT/forward_historical_tile",
    "_attention_from_completed": "RT/attention_reconstruction",
    "_historical_backward_tile": "RT/reverse_historical_tile",
}
VJP_PHASES = {
    (3, 1): "RT/local_writer_vjp",
    (1, 1): "RT/local_finish_vjp",
    (9, 7): "RT/batched_parameter_vjp",
}


def _is_rt_backward(frame):
    return (frame.f_globals.get("__name__") == olmo_tiled.__name__
            and frame.f_code.co_name == "backward")


def _arity(value):
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, (tuple, list)):
        return len(value)
    raise RuntimeError("Unexpected RT autograd.grad argument structure")


@contextmanager
def phase_annotations():
    """Single-thread diagnostic wrappers; no tensor operations are introduced.

    Only direct calls from native RT's custom backward get VJP/matmul labels.
    Other users of torch.autograd.grad, including checkpoint recomputation, pass
    through unchanged. Labels are inclusive and can nest; do not sum parents
    with children or add CPU-attributed device time to CUDA annotation spans.
    """
    counts = Counter()
    originals = {name: getattr(olmo_tiled, name) for name in HELPER_PHASES}
    original_grad, original_mm = torch.autograd.grad, olmo_tiled._mm

    def decorate(function, label):
        @wraps(function)
        def annotated(*args, **kwargs):
            counts[label] += 1
            with torch.profiler.record_function(label):
                return function(*args, **kwargs)
        return annotated

    @wraps(original_grad)
    def annotated_grad(*args, **kwargs):
        if not _is_rt_backward(sys._getframe(1)):
            return original_grad(*args, **kwargs)
        outputs = args[0] if args else kwargs.get("outputs")
        inputs = args[1] if len(args) > 1 else kwargs.get("inputs")
        signature = (_arity(outputs), _arity(inputs))
        if signature not in VJP_PHASES:
            raise RuntimeError(f"Unrecognized native RT VJP signature: {signature}")
        label = VJP_PHASES[signature]
        counts[label] += 1
        with torch.profiler.record_function(label):
            return original_grad(*args, **kwargs)

    @wraps(original_mm)
    def annotated_mm(*args, **kwargs):
        if not _is_rt_backward(sys._getframe(1)):
            return original_mm(*args, **kwargs)
        label = "RT/final_attention_matmul"
        counts[label] += 1
        with torch.profiler.record_function(label):
            return original_mm(*args, **kwargs)

    try:
        for name, function in originals.items():
            setattr(olmo_tiled, name, decorate(function, HELPER_PHASES[name]))
        torch.autograd.grad = annotated_grad
        olmo_tiled._mm = annotated_mm
        yield counts
    finally:
        torch.autograd.grad = original_grad
        olmo_tiled._mm = original_mm
        for name, function in originals.items():
            setattr(olmo_tiled, name, function)


def expected_backward_calls(length, recurrent_invocations=1):
    return {label: count * recurrent_invocations for label, count in {
        "RT/local_writer_vjp": length,
        "RT/local_finish_vjp": length,
        "RT/batched_parameter_vjp": 1,
        "RT/attention_reconstruction": 1,
        "RT/reverse_historical_tile": length - 1,
        # Global error, dQ, prefix dK and prefix dV. Surrounding pointwise math
        # is deliberately not attributed to this four-matmul annotation.
        "RT/final_attention_matmul": 4,
    }.items()}


def profile_rows(profile):
    rows = [{"name": event.key, "count": event.count,
             "cpu_total_us": event.cpu_time_total,
             "self_cpu_us": event.self_cpu_time_total,
             "device_total_us": event.device_time_total,
             "self_device_us": event.self_device_time_total,
             "device_type": str(event.device_type)}
            for event in profile.key_averages()]
    return sorted(rows, key=lambda row: row["self_device_us"], reverse=True)


def snapshot_backward(model, result):
    return {"losses": loss_snapshot(result),
            "gradients": {name: p.grad.detach().clone()
                          for name, p in model.named_parameters() if p.grad is not None}}


def check_observer_neutrality(model, result, reference, active_names):
    losses = loss_snapshot(result)
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    ownership = set(gradients) == set(reference["gradients"]) == set(active_names)
    loss_keys = set(losses) == set(reference["losses"])
    unequal_losses = sorted(name for name in set(losses) & set(reference["losses"])
                            if not torch.equal(losses[name], reference["losses"][name]))
    unequal_gradients = sorted(name for name in set(gradients) & set(reference["gradients"])
                               if not torch.equal(gradients[name], reference["gradients"][name]))
    return {"name": "observer_neutrality", "passed": ownership and loss_keys
            and not unequal_losses and not unequal_gradients,
            "ownership_matches": ownership, "loss_keys_match": loss_keys,
            "nonidentical_losses": unequal_losses,
            "nonidentical_gradients": unequal_gradients,
            "comparison": "bitwise same-state unannotated versus annotated eager backward"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("rt", "combined"), default="combined")
    parser.add_argument("--variant", choices=("reference", "triton"), default="reference")
    parser.add_argument("--batch-size", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path,
                        default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Imported here so the observer can be CPU-tested independently of the
    # native-checkpoint launcher and its integration source inventory.
    from scripts.olmo_f3c_validate import SOURCES as VALIDATION_SOURCES, new_plan
    from scripts.olmo_f1_common import build_model
    sources = tuple(sorted(set(VALIDATION_SOURCES) | {"scripts/olmo_f3c_profile.py"}))
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in sources:
        destination = args.output_dir / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    shutil.copyfile(PROTOCOL, args.output_dir / "protocol.md")
    report = {"schema": "olmo-f3c-profile-v1", "status": "running", "checks": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {k: str(v) for k, v in runtime.items()},
        "configuration": {"case": args.case, "variant": args.variant,
            "batch_size": args.batch_size, "length": 512, "rt_layers": [0],
            "checkpointing": True, "precision": "bf16_mixed",
            "ordinary_backend": "flash", "autocast_weight_cache": False,
            "cast_weights_once": True, "tile_backend": "triton",
            "backward_tile_backend": "eager" if args.variant == "reference" else "triton",
            **determinism},
        "source_hashes": {p: sha256_file(ROOT / p) for p in sources},
        "protocol_sha256": sha256_file(PROTOCOL),
        "optimizer_update_scope": {"eager": 3, "graph": 3, "total": 6,
            "meaning": "planned updates; completed count is recorded separately"},
        "profile_scope": {
            "timing": "Three complete graph updates before profiling; fixture creation excluded",
            "annotations": "Eager only; inclusive overlapping observer spans, not full-step clocks",
            "local_vjps": "Complete local autograd.grad calls; corresponding primals labeled separately",
            "final_attention_matmul": "Four global adjoint matmuls only; excludes surrounding pointwise work",
            "graph": "Unannotated replay kernel trace; kernels can overlap",
            "neutrality": "Bitwise all-active-gradient and loss check; no extra optimizer updates",
        }}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f3c-rt-backward", name="olmo-1b-" + args.output_dir.name)
    counters = TrainingCounters()

    def publish(check):
        report["checks"].append(check)
        write_json(args.output_dir / "report.json", report)
        if not check["passed"]:
            raise AssertionError(check["name"])

    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": report["configuration"], "checkpoint": report["checkpoint"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        case = case_for(args.case, batch=args.batch_size, length=512)
        with backend_context("flash"):
            model = build_model(state, case)
            plan = new_plan(model, changed_batch(tokenizer, case, 0), case.mode(), args.variant)
            optimizer, scheduler = build_optimizer(model)
            for step in range(3):
                plan.optimizer_step(optimizer, changed_batch(tokenizer, case, step),
                                    scheduler=scheduler, counters=counters)
            plan.capture(warmup=10)
            records = []
            batches = [changed_batch(tokenizer, case, 3 + i) for i in range(3)]

            def complete_timed():
                records.append(plan.optimizer_step(optimizer, batches[len(records)],
                    replay=True, scheduler=scheduler, counters=counters))

            report["full_step"] = timed(complete_timed, 3)
            report["input_tokens_per_second"] = plan.input_tokens / report["full_step"]["median_wall_seconds"]
            report["records"] = records
            # Memory belongs to the complete-update path, before temporary
            # all-gradient clones used by the observer-neutrality check.
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            report["current_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
            reference = snapshot_backward(model, plan.backward(replay=False))
            torch.cuda.synchronize()
            activity = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activity) as profile:
                with phase_annotations() as counts:
                    result = plan.backward(replay=False)
                torch.cuda.synchronize()
            report["annotated_eager_profile"] = profile_rows(profile)
            report["annotation_calls"] = dict(counts)
            del profile
            publish(check_observer_neutrality(model, result, reference, plan.active_names))
            del reference, result
            expected = expected_backward_calls(512)
            publish({"name": "expected_backward_annotations", "passed": all(
                counts[name] == value for name, value in expected.items()),
                "expected": expected, "actual": {name: counts[name] for name in expected}})
            with torch.profiler.profile(activities=activity) as profile:
                plan.backward(replay=True)
                torch.cuda.synchronize()
            report["graph_profile"] = profile_rows(profile)
            del profile
            report["health"] = state_health(model, optimizer)
            publish({"name": "finite_complete_updates", **report["health"]})
        assert report["source_hashes"] == {p: sha256_file(ROOT / p) for p in sources}
        assert report["protocol_sha256"] == sha256_file(PROTOCOL)
        tracker.log({"profile/input_tokens_per_second": report["input_tokens_per_second"],
            "profile/peak_allocated_gib": report["peak_allocated_gib"]}, step=6)
        report["status"] = "passed"
        print({"status": "passed", "tokens_per_second": report["input_tokens_per_second"],
               "annotation_calls": report["annotation_calls"]}, flush=True)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        report["completed_optimizer_updates"] = counters.optimizer_updates
        tracker.finish(succeeded=report["status"] == "passed")
        report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
