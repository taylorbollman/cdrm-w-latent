#!/usr/bin/env python3
"""Bounded 1/2/8-rank allocation measurements; never a production continuation.

Native runs clone an authenticated model and populated Adam state. They hold
the saved learning rate fixed, replay a declared ordered-data prefix, and use
fresh benchmark counters. Production checkpoints and their finite schedules
remain untouched. CUDA graphs and the campaign objective/reducer are reused
unchanged. Independent torchrun jobs require different output paths/rendezvous.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
from cdrm.pretrained.campaign_recipe import (
    CampaignRecipe, build_campaign_adamw, build_campaign_model, feedback_noise_for_rows,
)
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.distributed_checkpoint import inspect_distributed_checkpoint
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, optimizer_ownership, parameter_layout
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_campaign_loop import Coordinator
from scripts.olmo_campaign_probe import memory
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_packed_campaign_run import configure_cuda_runtime
from scripts.olmo_pilot_ordered_data import OrderedCampaignData
from scripts.olmo_two_gpu_validate import preserve_local_rng

SCHEMA = "olmo-allocation-benchmark-v1"


def allocation(rows, batch_size, world_size):
    if any(type(v) is not int or v <= 0 for v in (rows, batch_size, world_size)):
        raise ValueError("Rows, batch size and world size must be positive integers")
    slots = math.ceil(rows / (batch_size * world_size))
    return {"real_rows": rows, "world_size": world_size,
            "physical_batch_per_rank": batch_size, "microsteps_per_rank": slots,
            "physical_rows": slots * batch_size * world_size,
            "dummy_rows": slots * batch_size * world_size - rows}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("B", "NFR"), required=True)
    parser.add_argument("--scale", choices=("native", "tiny"), default="native")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--effective-rows", type=int, default=512)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measured-updates", type=int, default=4)
    parser.add_argument("--start-update", type=int, default=0,
                        help="Benchmark data prefix offset; not inherited training clock")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--index-sha256")
    parser.add_argument("--start-gate", type=Path,
                        help="External release JSON, created only after all jobs publish ready.json")
    parser.add_argument("--gate-id")
    parser.add_argument("--gate-timeout", type=float, default=1800.)
    parser.add_argument("--save-final-state", action="store_true", help="Tiny acceptance only")
    parser.add_argument("--wandb-project", default="pretrained-fbt-rt-nextlat")
    args = parser.parse_args(argv)
    for name in ("batch_size", "length", "effective_rows", "warmup_updates", "measured_updates"):
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if args.start_update < 0 or not 0 < args.gate_timeout <= 3600:
        parser.error("Require nonnegative data offset and bounded positive gate timeout")
    if args.warmup_updates > 8 or args.measured_updates > 12:
        parser.error("This disposable diagnostic is bounded to eight warmup and twelve measured updates")
    if bool(args.start_gate) != bool(args.gate_id):
        parser.error("Start gate and unique gate id must be supplied together")
    native_inputs = (args.checkpoint, args.checkpoint_sha256, args.corpus, args.index, args.index_sha256)
    if args.scale == "native":
        if any(value is None for value in native_inputs):
            parser.error("Native benchmarks require pinned checkpoint and ordered corpus/index")
        if (args.length, args.effective_rows) != (1024, 512) or args.save_final_state:
            parser.error("Native scope is T1024/effective512 and no disposable full checkpoint")
    elif any(value is not None for value in native_inputs) or args.length > 32:
        parser.error("Tiny fixtures use synthetic data, no native checkpoint, and T<=32")
    for pin in (args.checkpoint_sha256, args.index_sha256):
        if pin is not None and (len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin)):
            parser.error("SHA256 pins must be lowercase hexadecimal")
    args.output_dir = args.output_dir.absolute()
    if not args.output_dir.is_relative_to(ROOT) or args.output_dir.exists():
        parser.error("Output must be a new directory under the persistent project checkout")
    if args.start_gate is not None and args.start_gate.exists():
        parser.error("Start gate must not already exist")
    return args


def recipe_from_record(record):
    names = {field.name for field in fields(CampaignRecipe)}
    return CampaignRecipe(**{key: value for key, value in record.items() if key in names})


def validate_clone_payload(payload, manifest, model, optimizer):
    """Check byte-authenticated checkpoint ownership before applying any state.

    Deliberately does not restore rank RNG, cursor or scheduler: this is a new,
    disposable performance experiment, not rank-changing production recovery.
    """
    if payload["metadata"] != manifest["metadata"] or payload["counters"] != manifest["counters"]:
        raise ValueError("Checkpoint metadata/counters disagree with committed manifest")
    metadata = manifest["metadata"]
    if metadata["parameter_layout"] != parameter_layout(model):
        raise ValueError("Checkpoint model parameter layout differs")
    ownership = optimizer_ownership(model, optimizer)
    if metadata["optimizer_ownership"] != ownership:
        raise ValueError("Checkpoint optimizer ownership differs")
    current = model.state_dict()
    if current.keys() != payload["model"].keys():
        raise ValueError("Checkpoint model keys differ")
    for name, value in current.items():
        saved = payload["model"][name]
        if saved.shape != value.shape or saved.dtype != value.dtype:
            raise ValueError("Checkpoint tensor dimensions/dtype differ: " + name)
    for entry in metadata["parameter_layout"]:
        aliases = entry["aliases"]
        if any(not torch.equal(payload["model"][aliases[0]], payload["model"][name]) for name in aliases[1:]):
            raise ValueError("Checkpoint tied aliases differ")
    groups = payload["optimizer"]["param_groups"]
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("Checkpoint optimizer groups differ")
    identifiers = set()
    for old, new, names in zip(groups, optimizer.param_groups, ownership):
        if old.get("param_names") != names or len(old["params"]) != len(new["params"]):
            raise ValueError("Checkpoint Adam parameter order differs")
        for identifier, parameter in zip(old["params"], new["params"]):
            if identifier in identifiers:
                raise ValueError("Checkpoint Adam ownership duplicated")
            identifiers.add(identifier)
            state = payload["optimizer"]["state"].get(identifier)
            if not state or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Require populated standard Adam moments")
            if state["step"].numel() != 1 or state["step"].item() != manifest["counters"]["optimizer_updates"]:
                raise ValueError("Checkpoint Adam clock differs")
            for key in ("exp_avg", "exp_avg_sq"):
                if state[key].shape != parameter.shape or state[key].dtype != parameter.dtype:
                    raise ValueError("Checkpoint Adam moment dimensions/dtype differ")
    if identifiers != set(payload["optimizer"]["state"]):
        raise ValueError("Checkpoint Adam contains unowned state")


def construct_native(args, device):
    manifest = inspect_distributed_checkpoint(args.checkpoint,
        expected_manifest_sha256=args.checkpoint_sha256, verify_state=True)
    config = manifest["metadata"]["configuration"]
    recipe = recipe_from_record(config["recipe"])
    if recipe.arm != args.arm or recipe.sequence_length != 1024 or recipe.effective_valid_tokens != 524288:
        raise ValueError("Checkpoint recipe is outside the declared native benchmark")
    execution = config["execution_identity"]["payload"]["execution"]
    if execution["precision"] != "bf16_mixed" or execution["ordinary_attention"] != "flash_sdpa":
        raise ValueError("Native benchmark requires the accepted BF16/Flash path")
    payload = torch.load(args.checkpoint / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    base = OLMoTiledRTForCausalLM(OLMoConfig.from_dict(config["backbone"]), device="meta", dtype=torch.float32,
        attention_backend="sdpa", attention_precision=execution["rt_attention_precision"],
        ordinary_activation_checkpointing=execution["ordinary_activation_checkpointing"],
        cast_weights_once=execution["cast_weights_once"], tile_backend=execution["rt_forward_tiles"],
        backward_tile_backend=execution["rt_backward_tiles"], backward_memory=execution["backward_memory"],
        reuse_rope=execution["reuse_rope"], kv_only_writes=execution["kv_only_writes"],
        ordinary_pointwise_backend=execution["ordinary_pointwise_backend"],
        ordinary_rope_backend=execution["ordinary_rope_backend"])
    prefix = "backbone.backbone."
    base.load_state_dict({name[len(prefix):]: value for name, value in payload["model"].items()
                         if name.startswith(prefix)}, strict=True, assign=True)
    model = build_campaign_model(base, recipe)
    model.config = NextLatConfig(**config["model"])
    if model.predictor is not None:
        # Historically the predictor's own document policy can differ; policy
        # affects layout supplied by the outer model, not predictor arithmetic.
        from dataclasses import replace
        model.predictor.config = replace(model.predictor.config,
            lambda_kl=model.config.lambda_kl, lambda_latent=model.config.lambda_latent)
    optimizer = build_campaign_adamw(model, recipe, fused=True)
    validate_clone_payload(payload, manifest, model, optimizer)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).train()
    # Construct ownership after the move; do not retain optimizer references to
    # any parameters a device conversion might replace.
    optimizer = build_campaign_adamw(model, recipe, fused=True)
    optimizer.load_state_dict(payload["optimizer"])
    origin = {"manifest_sha256": manifest["manifest_sha256"], "state_sha256": manifest["state"]["sha256"],
              "original_world_size": manifest["world_size"], "original_counters": manifest["counters"],
              "learning_rates": [group["lr"] for group in optimizer.param_groups],
              "execution": execution, "configuration": config}
    state_stat = (args.checkpoint / "state.pt").stat()
    origin["state_stat"] = {"size": state_stat.st_size, "mtime_ns": state_stat.st_mtime_ns, "inode": state_stat.st_ino}
    del payload, base
    gc.collect()
    return model, optimizer, recipe, origin


def construct_tiny(args, device):
    recipe = CampaignRecipe(args.arm, sequence_length=args.length, rt_layers=(0, 1),
                            effective_valid_tokens=args.effective_rows * args.length,
                            document_policy="continuous-stream-v1")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261001)
        base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32",
            ordinary_activation_checkpointing=True, cast_weights_once=True,
            tile_backend="eager", backward_tile_backend="eager", backward_memory="recompute",
            reuse_rope=True, kv_only_writes=True)
        model = build_campaign_model(base, recipe).to(device).train()
    return model, build_campaign_adamw(model, recipe, fused=device.type == "cuda"), recipe, {
        "kind": "tiny-random-fresh-Adam", "seed": 20261001}


def tiny_batches(*, recipe, logical_update, rank, world_size, batch_size, effective_rows, width):
    allocation(effective_rows, batch_size, world_size)
    batches, noises, all_keys = [], [], []
    for start in range(0, effective_rows, batch_size * world_size):
        rows = range(start + rank * batch_size, min(start + (rank + 1) * batch_size, effective_rows))
        rows = list(rows)
        ids = torch.ones((batch_size, recipe.sequence_length), dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        docs = torch.full_like(ids, -1)
        keys = [f"allocation-{logical_update}-{row}" for row in rows]
        for local, row in enumerate(rows):
            ids[local] = (torch.arange(recipe.sequence_length) + row * 7 + logical_update * 3) % 59 + 2
            valid[local] = True
            docs[local] = row
        batches.append(NextLatBatch(ids, valid, docs))
        noises.append(feedback_noise_for_rows(recipe, keys, logical_update=logical_update,
            sequence_length=recipe.sequence_length, width=width, physical_batch_size=batch_size))
        all_keys.append(keys)
    return tuple(batches), tuple(noises), tuple(all_keys)


def wait_for_gate(args, coordinator, report):
    if args.start_gate is None:
        return
    def wait():
        write_json(args.output_dir / "ready.json", {"schema": SCHEMA, "gate_id": args.gate_id,
            "output_dir": str(args.output_dir), "world_size": coordinator.world_size, "ready_unix": time.time()})
        start = time.monotonic()
        while not args.start_gate.exists():
            if time.monotonic() - start > args.gate_timeout:
                raise TimeoutError("External benchmark start gate timed out")
            time.sleep(.2)
        gate = json.loads(args.start_gate.read_text())
        if gate.get("schema") != SCHEMA or gate.get("gate_id") != args.gate_id:
            raise ValueError("Benchmark gate identity differs")
        return {"path": str(args.start_gate), "sha256": sha256_file(args.start_gate), "release": gate}
    report["start_gate"] = coordinator.call("external measured-region gate", wait, rank_zero=True)


def summary_from_updates(updates):
    selected = [row for row in updates if row["phase"] == "measured"]
    if not selected:
        raise ValueError("No measured updates")
    useful = sum(row["metrics"]["input_tokens"] for row in selected)
    region = sum(max(rank["materialization"] + rank["backward"] + rank["optimizer"]
                     for rank in row["timing_by_rank"]) for row in selected)
    wall = selected[-1]["finished_unix"] - selected[0]["started_unix"]
    return {"measured_updates": len(selected), "real_input_tokens": useful,
            "selected_compute_materialization_seconds": region,
            "selected_compute_materialization_tokens_per_second": useful / region,
            "measured_window_seconds": wall, "measured_window_tokens_per_second": useful / wall,
            "measured_started_unix": selected[0]["started_unix"],
            "measured_finished_unix": selected[-1]["finished_unix"],
            "finite_updates": all(math.isfinite(row["metrics"]["gradient_norm_before_clip"])
                                  and math.isfinite(row["metrics"]["objective"]) for row in selected)}


def source_hashes():
    paths = set((ROOT / "cdrm/pretrained").rglob("*.py"))
    paths.update(ROOT / name for name in ("scripts/olmo_allocation_benchmark.py", "scripts/olmo_pilot_ordered_data.py",
        "scripts/olmo_campaign_loop.py", "scripts/olmo_campaign_probe.py", "scripts/olmo_distributed_prepare.py",
        "scripts/olmo_packed_campaign_run.py", "scripts/olmo_f2_graph_backend_probe.py", "scripts/olmo_validation.py",
        "scripts/olmo_two_gpu_validate.py", "scripts/experiment_tracking.py"))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}


def release_completed_graph_runner(runner, *, synchronize=None):
    """Release captured NCCL ownership before process-group destruction.

    Call only after a successfully completed, synchronized update boundary.
    A graph retains NCCL user objects even after its last replay; destroying
    the process group while these graphs remain reachable can wait forever.
    This wrapper owns shutdown only and does not modify the shared graph core.
    """
    if synchronize is None:
        synchronize = torch.cuda.synchronize
    synchronize(runner.device)
    for name in ("local_graph", "sync_graph"):
        graph = getattr(runner, name)
        if graph is not None:
            graph.reset()
        setattr(runner, name, None)
    runner.local_result = runner.sync_result = None
    runner.ddp = None
    runner.stream = None


def main(argv=None):
    args = parse_args(argv)
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Use the project Docker GPU launcher")
    world_size, local_rank = int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world_size not in (1, 2, 8):
        raise ValueError("This allocation scope supports one, two or eight ranks")
    if any(os.environ.get(key) != "0" for key in ("NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING")):
        raise RuntimeError("Both NCCL async-error flags must be zero for captured NCCL; bound the launcher externally")
    device, runtime, determinism = configure_cuda_runtime(local_rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600), device_id=device)
    coordinator = Coordinator()
    report = {"schema": SCHEMA, "status": "running", "arm": args.arm, "scale": args.scale,
              "scope": "Disposable allocation benchmark; no training continuation, quality or precision clearance",
              "allocation": allocation(args.effective_rows, args.batch_size, world_size),
              "length": args.length, "sources": source_hashes(), "runtime": runtime,
              "determinism": determinism, "updates": [], "data_start_update": args.start_update,
              "warmup_updates": args.warmup_updates, "measured_updates": args.measured_updates,
              "scheduler": "Held at saved optimizer LR; no inherited finite scheduler is extended",
              "checkpoint_policy": "Source immutable; disposable clone updates not retained; per-update evidence persists"}
    tracker = None
    data = None
    started = time.monotonic()
    def persist():
        report["wandb"] = tracker.record if tracker else None
        write_json(args.output_dir / "report.json", report)
    def setup():
        nonlocal tracker
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, digest in report["sources"].items():
            destination = args.output_dir / "source-snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
            if sha256_file(destination) != digest:
                raise ValueError("Source changed during snapshot")
        tracker = OnlineTracker(project=args.wandb_project, output_dir=args.output_dir,
            name=args.output_dir.name, group="olmo-allocation-readiness", preserve_state=preserve_local_rng)
        tracker.start({key: report[key] for key in ("scope", "arm", "scale", "allocation", "length")})
        persist()
    try:
        coordinator.call("benchmark output and tracking", setup, rank_zero=True)
        model, optimizer, recipe, origin = coordinator.call("construct performance clone",
            lambda: construct_native(args, device) if args.scale == "native" else construct_tiny(args, device))
        report["origin"] = origin
        report["recipe"] = recipe.to_dict()
        report["objective_weights"] = model.objective_weights()
        report["recipe"]["optimizer_state"] = "inherited_performance_clone" if args.scale == "native" else "fresh"
        report["recipe"]["auxiliary"]["kl"] = model.config.lambda_kl
        report["recipe"]["auxiliary"]["latent"] = model.config.lambda_latent
        report["parameters"] = {"resident": sum(p.numel() for p in model.parameters()),
                                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if args.scale == "native":
            data = coordinator.call("ordered benchmark data", lambda: OrderedCampaignData(args.corpus, args.index))
            if data.manifest_sha256 != args.index_sha256 or data.length != args.length or data.split != "train":
                raise ValueError("Ordered benchmark authority differs")
            for _ in range(args.start_update):
                plan = data.peek_update(data.cursor(), args.effective_rows * args.length)
                data.commit(plan.start_cursor, plan)
            report["data"] = {"index_sha256": data.manifest_sha256, "start_cursor": asdict(data.cursor()),
                              "corpus": str(args.corpus), "index": str(args.index)}
        def materialize(index):
            if data is None:
                batches, noises, keys = tiny_batches(recipe=recipe, logical_update=args.start_update + index,
                    rank=coordinator.rank, world_size=world_size, batch_size=args.batch_size,
                    effective_rows=args.effective_rows, width=model.config.model_dim)
                return batches, noises, keys, None
            plan = data.peek_update(data.cursor(), args.effective_rows * args.length)
            if plan is None or len(plan.rows) != args.effective_rows or plan.counts.valid_tokens != args.effective_rows * args.length:
                raise ValueError("Benchmark data exhausted or effective update differs")
            packed = data.rank_batches(plan, rank=coordinator.rank, world_size=world_size, physical_batch_size=args.batch_size)
            noises = tuple(feedback_noise_for_rows(recipe, keys, logical_update=plan.start_cursor.next_update,
                sequence_length=args.length, width=model.config.model_dim, physical_batch_size=args.batch_size)
                for keys in packed.keys)
            return packed.batches, noises, packed.keys, plan
        training = LMTrainingConfig(precision="bf16_mixed" if args.scale == "native" else "fp32",
                                    max_grad_norm=recipe.max_grad_norm)
        context = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if args.scale == "native" else nullcontext()
        with disable_autocast_weight_cache(), context:
            batches, noises, keys, _ = coordinator.call("preparation data", lambda: materialize(0))
            counts = sum_objective_counts(coordinator.gather(sum_objective_counts([model.counts(b) for b in batches])))
            adapter = CampaignObjective(model, batches[0], mode=recipe.mode(), global_counts=counts,
                world_size=world_size, feedback_noise=noises[0], config=training)
            runner = CampaignDDPGraphTraining(adapter)
            preparation = time.monotonic()
            runner.prepare(warmup=11)
            runner.capture(warmup=11, release_transient_cache=True)
            report["preparation_seconds_by_rank"] = coordinator.gather(time.monotonic() - preparation)
            report["memory_after_capture_by_rank"] = coordinator.gather(memory())
            del batches, noises, keys
            counters = TrainingCounters()
            for index in range(args.warmup_updates + args.measured_updates):
                if index == args.warmup_updates:
                    wait_for_gate(args, coordinator, report)
                    torch.cuda.reset_peak_memory_stats(device)
                dist.barrier()
                torch.cuda.synchronize(device)
                started_unix = time.time()
                step_started = time.perf_counter()
                batches, noises, keys, plan = coordinator.call("update materialization", lambda: materialize(index))
                materialization_seconds = time.perf_counter() - step_started
                torch.cuda.synchronize(device)
                backward_started = time.perf_counter()
                result = runner.backward(batches, feedback_noises=noises, replay=True)
                torch.cuda.synchronize(device)
                backward_seconds = time.perf_counter() - backward_started
                optimizer_started = time.perf_counter()
                metrics = runner.step(result, optimizer, counters=counters)
                if data is not None:
                    data.commit(plan.start_cursor, plan)
                torch.cuda.synchronize(device)
                optimizer_seconds = time.perf_counter() - optimizer_started
                if metrics["input_tokens"] != args.effective_rows * args.length:
                    raise ValueError("Loss-masked dummy rows changed real token accounting")
                coordinator.same("globally reduced update metrics", metrics)
                timing = coordinator.gather({"materialization": materialization_seconds,
                    "backward": backward_seconds, "optimizer": optimizer_seconds})
                key_digest = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
                row = {"update": index + 1, "phase": "warmup" if index < args.warmup_updates else "measured",
                       "metrics": metrics, "timing_by_rank": timing, "memory_by_rank": coordinator.gather(memory()),
                       "row_key_sha256_by_rank": coordinator.gather(key_digest),
                       "started_unix": min(coordinator.gather(started_unix)), "finished_unix": time.time()}
                report["updates"].append(row)
                def log():
                    persist()
                    seconds = max(sum(t.values()) for t in timing)
                    tracker.log({"update": index + 1, "benchmark/tokens_per_second": metrics["input_tokens"] / seconds,
                        "benchmark/gradient_norm": metrics["gradient_norm_before_clip"],
                        "benchmark/objective": metrics["objective"], "benchmark/measured": row["phase"] == "measured"},
                        step=index + 1)
                coordinator.call("completed update evidence", log, rank_zero=True)
                del batches, noises, keys, plan
            report["summary"] = summary_from_updates(report["updates"])
            report["runner_by_rank"] = coordinator.gather(runner.metadata)
            report["final_counters"] = asdict(counters)
            if args.save_final_state:
                def save_tiny():
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "counters": asdict(counters)}, args.output_dir / "tiny-final-state.pt")
                    return sha256_file(args.output_dir / "tiny-final-state.pt")
                report["tiny_final_state_sha256"] = coordinator.call("tiny final state", save_tiny, rank_zero=True)
        if source_hashes() != report["sources"]:
            raise ValueError("Runtime sources changed during benchmark")
        if args.scale == "native":
            if sha256_file(args.checkpoint / "manifest.json") != args.checkpoint_sha256:
                raise ValueError("Original checkpoint manifest changed")
            state_stat = (args.checkpoint / "state.pt").stat()
            if origin["state_stat"] != {"size": state_stat.st_size, "mtime_ns": state_stat.st_mtime_ns, "inode": state_stat.st_ino}:
                raise ValueError("Original checkpoint state-file metadata changed")
            report["original_checkpoint_unchanged"] = True
        report["status"] = "updates_complete"
        coordinator.call("completed summary", lambda: tracker.summary(report["summary"]), rank_zero=True)
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                              "traceback": traceback.format_exc()})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        completed_work = report["status"] == "updates_complete"
        tracking_error = None
        try:
            if coordinator.rank == 0 and args.output_dir.exists():
                persist()
                if tracker:
                    tracker.finish(succeeded=completed_work)
                    persist()
        except BaseException as error:
            tracking_error = error
            report.update(status="failed", error={"type": type(error).__name__,
                "message": str(error), "scope": "final tracking synchronization"})
        if data is not None:
            data.close()
        if completed_work:
            try:
                report["teardown"] = "releasing_graphs_and_reducer"
                if coordinator.rank == 0:
                    persist()
                release_completed_graph_runner(runner)
                # Drop all remaining references before collecting any reducer
                # cycles. The historical executors achieve this by returning
                # from their segment function before process-group shutdown.
                runner = adapter = optimizer = model = None
                gc.collect()
                report["teardown"] = "destroying_process_group"
                if coordinator.rank == 0:
                    persist()
                dist.destroy_process_group()
                report["teardown"] = "completed"
                if tracking_error is None:
                    report["status"] = "completed"
            except BaseException as error:
                report.update(status="failed", error={"type": type(error).__name__,
                    "message": str(error), "scope": "completed-boundary teardown"})
                if coordinator.rank == 0:
                    persist()
                raise
        # Following an unknown CUDA/NCCL failure, avoid inventing additional
        # collectives or synchronized CUDA cleanup; the supervisor terminates
        # the disposable job. Partial per-update evidence is already durable.
        if coordinator.rank == 0 and args.output_dir.exists():
            report["elapsed_seconds"] = time.monotonic() - started
            persist()
        if tracking_error is not None:
            raise tracking_error


if __name__ == "__main__":
    main()
