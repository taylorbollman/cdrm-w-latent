#!/usr/bin/env python3
"""Bounded random ordinary-OLMo throughput; no production math changes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTMode
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig, _chunked_sum, _ce_chunk
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.static_training import StaticFBTTraining
from scripts.olmo_f1_common import build_optimizer, state_health
from scripts.olmo_f3_graph_training import timed, backend_context, configure_determinism
from scripts.olmo_f4_resources import SOURCES as F4_SOURCES, memory_snapshot
from scripts.olmo_validation import require_container_gpu
from scripts.experiment_tracking import OnlineTracker

PROTOCOL = ROOT / "docs/reports/olmo-ordinary-throughput/protocol.md"
SOURCES = tuple(sorted(set(F4_SOURCES) | {"scripts/olmo_ordinary_throughput.py"}))


def make_config(layers=6, heads=32, mlp=8192):
    return OLMoConfig(num_layers=layers, num_heads=heads, mlp_intermediate_size=mlp)


def make_batch(batch_size, length, seed=20260923, supervision="half", vocab_size=50304):
    if batch_size < 1 or length < 2 or vocab_size < 2 or supervision not in ("half", "full"):
        raise ValueError("Invalid random throughput fixture")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(vocab_size, (batch_size, length), generator=generator)
    valid = torch.ones_like(ids, dtype=torch.bool)
    docs = torch.arange(batch_size)[:, None].expand_as(ids).clone()
    mask = valid.clone()
    if supervision == "half":
        mask[:, :length // 2] = False
    return NextLatBatch(ids, valid, docs, ce_mask=mask, latent_mask=valid.clone(), kl_mask=mask.clone())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, choices=(1, 32, 64, 128, 256, 512), required=True)
    p.add_argument("--layers", type=int, choices=(6, 16), default=6)
    p.add_argument("--heads", type=int, choices=(16, 32), default=32)
    p.add_argument("--mlp", type=int, choices=(4096, 8192), default=8192)
    p.add_argument("--length", type=int, choices=(32, 512), default=512)
    p.add_argument("--loss-chunk", type=int, choices=(128, 512, 2048, 4096), default=128)
    p.add_argument("--supervision", choices=("half", "full"), default="half")
    p.add_argument("--checkpointing", choices=("on", "off"), default="on")
    p.add_argument("--execution", choices=("graph", "eager"), default="graph")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if args.profile and args.batch_size > 64:
        p.error("Operator profiling is bounded to B64")
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(20260923)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        dest = args.output_dir / "source-snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    shutil.copyfile(PROTOCOL, args.output_dir / "protocol.md")
    config = make_config(args.layers, args.heads, args.mlp)
    configuration = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": config.to_dict(), "initialization": "random native Mitchell", "seed": 20260923,
        "precision": "bf16_mixed", "parameter_optimizer_dtype": "float32",
        "rt": False, "fbt": False, "nextlat": False, "attention": "deterministic pytorch Flash",
        "physical_batch": args.batch_size, "accumulation_steps": 1, "world_size": 1,
        "autocast_weight_cache": False, "tf32": False, "warmup_updates": 3,
        "capture_warmup_backwards": 10, "timed_updates": 3,
        "loss_chunk_axis": "selected positions, each projected over the full vocabulary"}
    report = {"schema": "olmo-ordinary-throughput-v1", "status": "running", "stage": "initialization",
        "started_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "runtime": runtime, "determinism": determinism,
        "source_hashes": {name: sha256_file(ROOT/name) for name in SOURCES},
        "protocol_sha256": sha256_file(PROTOCOL), "physical_optimizer_updates": 0}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-ordinary-throughput",
                            name=args.output_dir.name, output_dir=args.output_dir)

    def save(stage=None):
        if stage is not None:
            report["stage"] = stage
            print({"stage": stage, "utc": datetime.now(timezone.utc).isoformat()}, flush=True)
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)

    try:
        tracker.start(configuration)
        print({"wandb": tracker.record["run_url"]}, flush=True)
        save()
        base = OLMoTiledRTForCausalLM(config, ordinary_activation_checkpointing=args.checkpointing == "on",
                                     device="cuda", dtype=torch.float32)
        core = OLMoFBT(base)
        core.fusion.requires_grad_(False)
        model = FBTNextLatLM(core, NextLatConfig(model_dim=config.model_dim,
                            vocab_chunk_size=args.loss_chunk), enabled=False).train()
        weight_probe = base.layers[0].ff_out.weight.detach().flatten()[:4096].clone()
        report["parameters"] = {"active": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                "resident": sum(p.numel() for p in model.parameters()),
                                "frozen_fusion": sum(p.numel() for p in core.fusion.parameters())}
        batch = make_batch(args.batch_size, args.length, supervision=args.supervision)
        mode = FBTMode(enabled=False, num_passes=1)
        with backend_context("flash"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                hidden_probe = torch.randn(2048, config.model_dim, device="cuda") * .2
                target_probe = torch.arange(2048, device="cuda") % config.vocab_size
                losses = {chunk: float(_chunked_sum(_ce_chunk, hidden_probe, base.readout_weight,
                    target_probe, chunk, weight_second=True) / 2048) for chunk in (128, 2048)}
            delta = abs(losses[128] - losses[2048]) / max(abs(losses[128]), 1e-30)
            report["same_weight_ce_chunk_check"] = {"losses": losses, "relative_difference": delta,
                "passed": delta < 1e-5, "scope": "2048 native-width random hidden positions; same full readout; forward only"}
            if not report["same_weight_ce_chunk_check"]["passed"]:
                raise AssertionError("Loss-chunk regrouping changed the bounded CE check")
            del hidden_probe, target_probe
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    positions = torch.arange(32, device="cuda")[None, :]
                    base.layers[0](torch.randn(1, 32, config.model_dim, device="cuda"), past=None,
                        query_positions=positions, key_positions=positions, mask=None,
                        is_causal=True, attention_backend="sdpa")
            report["ordinary_dispatch"] = sorted({e.key for e in profile.key_averages()
                                                  if "scaled_dot_product" in e.key})
            if not any("flash_attention" in name for name in report["ordinary_dispatch"]):
                raise AssertionError("Ordinary attention did not dispatch to Flash")
            plan = StaticFBTTraining(model, batch, mode=mode, config=LMTrainingConfig(precision="bf16_mixed"))
            optimizer, scheduler = build_optimizer(model)
            counters = TrainingCounters()
            report["counts"] = dict(plan.counts)
            report["input_tokens"] = plan.input_tokens
            report["ce_chunks"] = (plan.counts["ce"] + args.loss_chunk - 1) // args.loss_chunk
            torch.cuda.reset_peak_memory_stats()
            save("eager_preparation")
            preparation = []
            for i in range(3):
                fresh = make_batch(args.batch_size, args.length, seed=20261000+i, supervision=args.supervision)
                preparation.append(plan.optimizer_step(optimizer, fresh, scheduler=scheduler, counters=counters))
                report["physical_optimizer_updates"] = counters.optimizer_updates
                report["preparation_records"] = preparation
                save()
                print({"prepared_update": i+1, "loss": preparation[-1]["objective"]}, flush=True)
            if args.execution == "graph":
                save("capture_warmup")
                started = time.perf_counter()
                plan.capture(warmup=10)
                report["setup_capture_seconds"] = time.perf_counter() - started
            else:
                save("eager_backward_warmup")
                for _ in range(10):
                    plan.backward(replay=False)
                torch.cuda.synchronize()
            report["setup_memory"] = memory_snapshot()
            save("timed_complete_updates")
            torch.cuda.reset_peak_memory_stats()
            records = []
            timed_batches = [make_batch(args.batch_size, args.length, seed=20261100+i,
                                       supervision=args.supervision) for i in range(3)]

            def update():
                records.append(plan.optimizer_step(optimizer, timed_batches[len(records)], replay=args.execution == "graph",
                                                   scheduler=scheduler, counters=counters))

            report["full_update"] = timed(update, 3)
            report["physical_optimizer_updates"] = counters.optimizer_updates
            report["timed_records"] = records
            report["steady_memory"] = memory_snapshot()
            report["input_tokens_per_second"] = plan.input_tokens / report["full_update"]["median_wall_seconds"]
            save("timed_backward_only")
            report["forward_loss_backward"] = timed(lambda: plan.backward(replay=args.execution == "graph"), 3)
            report["forward_loss_backward_tokens_per_second"] = plan.input_tokens / report["forward_loss_backward"]["median_wall_seconds"]
            report["health"] = state_health(model, optimizer)
            report["trainable_weight_changed"] = not torch.equal(weight_probe,
                base.layers[0].ff_out.weight.detach().flatten()[:4096])
            report["gradient_health"] = {"missing": [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None],
                "nonfinite": [n for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]}
            if not report["health"]["passed"] or any(report["gradient_health"].values()) or not report["trainable_weight_changed"]:
                raise AssertionError("Nonfinite or missing participating state")
            if args.profile:
                save("untimed_operator_profile")
                import gzip
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    plan.backward(replay=args.execution == "graph")
                    torch.cuda.synchronize()
                path = args.output_dir / "operator-trace.json"
                profile.export_chrome_trace(str(path))
                with path.open("rb") as src, gzip.open(path.with_suffix(".json.gz"), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                path.unlink()
                events = [e for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                totals = {}
                for e in events:
                    row = totals.setdefault(e.name, {"calls": 0, "self_device_us": 0.0})
                    row["calls"] += 1
                    row["self_device_us"] += e.self_device_time_total
                report["profile"] = {"scope": "untimed forward/loss/backward; excludes optimizer",
                    "device_event_count": len(events), "device_kernels": totals,
                    "trace_sha256": sha256_file(args.output_dir / "operator-trace.json.gz")}
            if any(sha256_file(ROOT/name) != digest for name,digest in report["source_hashes"].items()):
                raise AssertionError("A recorded runtime source changed during benchmark")
            report["status"] = "passed"
            tracker.log({"benchmark/input_tokens_per_second": report["input_tokens_per_second"],
                "benchmark/forward_loss_backward_tokens_per_second": report["forward_loss_backward_tokens_per_second"],
                "benchmark/setup_peak_reserved_gib": report["setup_memory"]["peak_reserved_gib"],
                "benchmark/steady_peak_allocated_gib": report["steady_memory"]["peak_allocated_gib"]})
            tracker.summary({"result_status": report["status"], "input_tokens_per_second": report["input_tokens_per_second"],
                             "physical_optimizer_updates": counters.optimizer_updates})
            save("complete")
            print({"status": report["status"], "input_tokens_per_second": report["input_tokens_per_second"],
                   "setup_memory": report["setup_memory"]}, flush=True)
    except BaseException as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
                      error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        save()
        raise
    finally:
        tracker.finish(succeeded=report["status"] == "passed")
        save()


if __name__ == "__main__":
    main()
