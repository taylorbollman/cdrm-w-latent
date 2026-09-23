#!/usr/bin/env python3
"""Eight-way native feature resource comparisons with bounded graph validation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import TrainingCounters
from scripts.olmo_f1_common import IntegrationCase
from scripts.olmo_f3d_validate import new_plan
from scripts.olmo_f3_graph_training import (
    changed_batch, build_optimizer, build_model, state_health,
    compare_graph, full_update_parity, timed, configure_determinism,
    require_container_gpu, validate_prepared_manifest, load_native_state_dict,
    load_native_tokenizer, backend_context, OnlineTracker,
)

from scripts.olmo_f3e_validate import (SOURCES as F3E_SOURCES, compare_variant, resource_card)

# Independent RT / FBT / NextLat switches; no case-name inference elsewhere.
FEATURES = {
    "ordinary": (False, False, False), "rt": (True, False, False),
    "fbt": (False, True, False), "nextlat": (False, False, True),
    "rt-fbt": (True, True, False), "rt-nextlat": (True, False, True),
    "fbt-nextlat": (False, True, True), "combined": (True, True, True),
}
SOURCES = tuple(sorted(set(F3E_SOURCES) | {"scripts/olmo_f4_resources.py"}))
PROTOCOL = ROOT / "docs/reports/olmo1b-f4/protocol.md"


def selected_case(name, *, batch, length):
    if name not in FEATURES:
        raise ValueError("Unknown F4 feature combination")
    rt, fbt, nextlat = FEATURES[name]
    return IntegrationCase(name, rt_layers=(0, 15) if rt else (), fbt=fbt,
                           nextlat=nextlat, batch_size=batch, length=length)


def memory_snapshot():
    return {"allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


def operator_trace(plan, directory):
    """Untimed eager backward trace: coverage audit, not hardware FLOP truth."""
    import gzip
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as profile:
        plan.backward(replay=False)
        torch.cuda.synchronize()
    path = directory / "operator-trace.json"
    profile.export_chrome_trace(str(path))
    compressed = path.with_suffix(".json.gz")
    with path.open("rb") as source, gzip.open(compressed, "wb") as target:
        shutil.copyfileobj(source, target)
    path.unlink()
    rows = [{"name": event.key, "calls": event.count, "estimated_flops": event.flops,
             "self_cpu_time_us": event.self_cpu_time_total,
             "self_device_time_us": event.self_device_time_total}
            for event in profile.key_averages()]
    device_events = [event for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    return {"file": compressed.name, "sha256": sha256_file(compressed), "bytes": compressed.stat().st_size,
            "operator_rows": rows, "observed_device_event_count": len(device_events),
            "observed_device_event_names": sorted({event.name for event in device_events}),
            "pytorch_estimated_flops": sum(row["estimated_flops"] for row in rows),
            "scope": "One eager forward/loss/backward after correctness; no optimizer. Shapes and selected-op FLOPs only. Fused Flash/Triton, checkpointing and custom backward can be undercounted; do not use this as measured complete-update FLOPs or timing."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(FEATURES), required=True)
    parser.add_argument("--variant", choices=("recompute",), default="recompute")
    parser.add_argument("--stage", choices=("correctness", "capacity"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, choices=(32, 512), default=32)
    parser.add_argument("--operator-trace", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 96:
        parser.error("bounded batch 1..96")
    if args.stage == "capacity" and (args.length != 512 or args.batch_size not in (64, 96)):
        parser.error("F4 capacity uses common B64/T512 or bounded B96/T512")
    if args.operator_trace and (args.stage != "correctness" or args.batch_size != 1 or args.length != 32):
        parser.error("trace is separate from timed capacity and bounded at B1/T32")
    return args


def main(argv=None):
    args = parse_args(argv)
    case = selected_case(args.case, batch=args.batch_size, length=args.length)
    mode = case.mode()
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        target = args.output_dir / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(PROTOCOL, args.output_dir / "protocol.md")
    configuration = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "forward_tile_backend": "triton", "cast_weights_once": True,
        "backward_tile_backend": "triton",
        "backward_memory": "recompute" if args.variant == "recompute" else "materialized",
        "case_specification": asdict(case), "mode": asdict(mode),
        "selected_rt_layers": list(case.rt_layers), "layout": "spread2" if case.rt_layers else "none",
        "feature_flags": {"rt": bool(case.rt_layers), "fbt": case.fbt, "nextlat": case.nextlat},
        "world_size": 1, "accumulation_steps": 1, "physical_batch_per_gpu": case.batch_size, "logical_batch": case.batch_size, "layout_role": "common_feature_comparison",
        "rt_block_calls_per_forward_backward": len(case.rt_layers) * (mode.num_passes - 1 if mode.enabled else 1),
        "ordinary_activation_checkpointing": True, "ordinary_attention_backend": "deterministic_flash",
        "precision": "bf16_mixed", "tf32": False, "autocast_weight_cache": False}
    report = {"schema": "olmo-f4-native-v1", "status": "running", "stage": args.stage,
        "configuration": configuration, "runtime": {k: str(v) for k, v in runtime.items()},
        "determinism": determinism, "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
        "source_hashes": {p: sha256_file(ROOT / p) for p in SOURCES}, "protocol_sha256": sha256_file(PROTOCOL)}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f4-features", name="olmo-" + args.output_dir.name)

    def publish(check):
        if check["name"].startswith("candidate_"):
            check["passed"] = check["passed"] and check["all_bitwise_equal"]
        report["checks"].append(check)
        write_json(args.output_dir / "report.json", report)
        print({"check": check["name"], "passed": check["passed"], "exact": check.get("all_bitwise_equal")}, flush=True)
        tracker.log({"correctness/passed": int(check["passed"])}, step=len(report["checks"]))
        if not check["passed"]:
            raise AssertionError(check["name"])

    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": configuration, "checkpoint": report["checkpoint"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        model = build_model(state, case)
        batch = changed_batch(tokenizer, case, 0)
        with backend_context("flash"):
            if args.stage == "correctness":
                plan, comparison = compare_variant(model, batch, mode, args.variant)
                publish(comparison)
                plan.capture(warmup=10)
                publish(compare_graph(plan, "candidate_initial_graph"))
                plan.load_batch(changed_batch(tokenizer, case, 1))
                publish(compare_graph(plan, "candidate_changed_tokens_and_overwrite", replays=2))
                publish(full_update_parity(plan, tokenizer, case, updates=3))
                plan.load_batch(changed_batch(tokenizer, case, 2))
                publish(compare_graph(plan, "candidate_changed_weights"))
                report["resources"] = resource_card(plan, case)
            else:
                plan = new_plan(model, batch, mode, args.variant)
                optimizer, scheduler = build_optimizer(model)
                torch.cuda.reset_peak_memory_stats()
                counters = TrainingCounters()
                for step in range(3):
                    plan.optimizer_step(optimizer, changed_batch(tokenizer, case, step), scheduler=scheduler, counters=counters)
                plan.capture(warmup=10)
                torch.cuda.synchronize()
                setup_memory = memory_snapshot()
                torch.cuda.reset_peak_memory_stats()
                batches = [changed_batch(tokenizer, case, i + 3) for i in range(3)]
                records = []

                def complete():
                    records.append(plan.optimizer_step(optimizer, batches[len(records)], replay=True,
                                                       scheduler=scheduler, counters=counters))

                timing = timed(complete, 3)
                steady_memory = memory_snapshot()
                resources = resource_card(plan, case, optimizer=optimizer)
                seconds = timing["median_wall_seconds"]
                analytic = resources["analytic_matrix_work"]
                report["resources"] = resources
                report["capacity"] = {"full_step": timing, "input_tokens_per_second": plan.input_tokens / seconds,
                    "ce_targets_per_second": plan.counts["ce"] / seconds,
                    "estimated_matrix_tflops_per_second_minimum": analytic["matrix_flops_minimum"] / seconds / 1e12,
                    "estimated_matrix_tflops_per_second_maximum": analytic["matrix_flops_maximum"] / seconds / 1e12,
                    "records": records, "warmup_updates": 3, "timed_updates": 3,
                    "physical_optimizer_updates": 6, "backward_only_warmup": 10,
                    "peak_allocated_gib": max(setup_memory["peak_allocated_gib"], steady_memory["peak_allocated_gib"]),
                    "peak_reserved_gib": max(setup_memory["peak_reserved_gib"], steady_memory["peak_reserved_gib"]),
                    "setup_memory": setup_memory, "steady_memory": steady_memory,
                    "current_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "health": state_health(model, optimizer),
                    "scope": "Input copy + graph forward/loss/backward + clip + AdamW + scheduler; one physical batch."}
                publish({"name": "finite_complete_updates", "passed": report["capacity"]["health"]["passed"]})
                tracker.log({"capacity/input_tokens_per_second": report["capacity"]["input_tokens_per_second"],
                    "capacity/ce_targets_per_second": report["capacity"]["ce_targets_per_second"],
                    "capacity/seconds_per_update": seconds,
                    "capacity/peak_reserved_gib": report["capacity"]["peak_reserved_gib"],
                    "capacity/peak_allocated_gib": report["capacity"]["peak_allocated_gib"]}, step=len(report["checks"]) + 1)
                print(report["capacity"], flush=True)
            if args.operator_trace:
                report["operator_trace"] = operator_trace(plan, args.output_dir)
            report["backward_preparation"] = {"warmup": plan.warmup_backward_calls,
                "capture": plan.capture_backward_calls, "replay": plan.replay_calls}
            report["memory"] = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "current_reserved_gib": torch.cuda.memory_reserved() / 2**30}
        assert report["source_hashes"] == {p: sha256_file(ROOT / p) for p in SOURCES}
        assert report["protocol_sha256"] == sha256_file(PROTOCOL)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        tracker.finish(succeeded=report["status"] == "passed")
        report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
