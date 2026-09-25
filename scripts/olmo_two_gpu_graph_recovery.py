#!/usr/bin/env python3
"""Two-H100 graph teardown/checkpoint/reconstruction continuation diagnostic.

This does not save through a live graph. We release graph and reducer hooks,
clear gradients, and save the boundary. Reference and restored continuations
each build a new actual-DDP graph. Checkpoints remain on persistent storage for
separate verified GCS retention. Launch with NCCL async-error handling disabled
and an external timeout, matching the distributed graph validation harness.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import gc
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.ddp_graph_training import PreparedDDPObjective, DDPGraphTraining
from cdrm.pretrained.distributed_checkpoint import _local_rng, save_distributed_checkpoint, load_distributed_checkpoint
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, TERMS
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import IntegrationCase, active_names, state_health
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_rt_large_batch import (
    OnlineTracker, backend_context, compiler_configuration, configure_determinism,
    dependency_record, finish_tracking, load_native_tokenizer, optimizer_for,
    require_container_gpu, validate_prepared_manifest,
)
from scripts.olmo_two_gpu_graph import fixed_batch, finish_step
from scripts.olmo_two_gpu_recovery import (
    PROTOCOL, coordinated, draw_rng, ensure_persistent_checkpoint_directory,
    execution_configuration, local_boundary, parse_args, seed_local, utc_now,
)
from scripts.olmo_two_gpu_validate import construct, gather, move_batch, preserve_local_rng


def graph_cursor(case, rank, update):
    return {"schema": "olmo-two-gpu-graph-recovery-cursor-v1", "rank": rank, "world_size": 2,
            "next_update": update, "microbatches_per_rank": 1,
            "batch_size_per_rank": case.batch_size, "length": case.length,
            "batch_generator": "olmo_two_gpu_graph.fixed_batch-v1"}


def validate_graph_cursor(cursor, case, rank, counters):
    if cursor != graph_cursor(case, rank, counters.optimizer_updates):
        raise ValueError("Graph recovery cursor differs from counters/case/rank")


def release_graph_and_hooks(runtime):
    """Caller next drops the runtime reference, collects, then clears gradients.

    Explicitly remove the old reducer's hooks before rewrapping the same model.
    This private API is pinned to the installed PyTorch source in the report.
    No old graph may replay after this operation.
    """
    remove_hooks = getattr(runtime.ddp, "_remove_autograd_hooks", None)
    if not callable(remove_hooks):
        raise RuntimeError("Installed DDP lacks the required explicit autograd-hook teardown API")
    torch.cuda.synchronize(runtime.device)
    runtime.graph_result = None
    runtime.graph = None
    remove_hooks()
    runtime.ddp = None
    runtime.stream = None


def graph_configuration(model, case, config, args, counts_by_rank, expected_active):
    result = execution_configuration(model, case, config, args)
    result.update(schema="olmo-two-gpu-graph-recovery-execution-v1", graphs=True,
                  recovery_scope="both_continuations_rebuild_actual_ddp_graphs",
                  batch_generator="olmo_two_gpu_graph.fixed_batch-v1",
                  microbatches_per_rank=1,
                  counts_by_rank=counts_by_rank,
                  global_counts={term: sum(counts[term] for counts in counts_by_rank) for term in TERMS},
                  expected_active_names=list(expected_active), warmup_backward_calls_per_graph=11,
                  ddp={"find_unused_parameters": False, "gradient_as_bucket_view": False,
                       "broadcast_buffers": False, "static_graph": True, "bucket_cap_mb": 25})
    return result


def run(args, report, tracker, persist, publish):
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    case = IntegrationCase(args.case, fbt=args.case == "combined", nextlat=args.case == "combined",
        rt_layers=(0, 1 if args.tiny else 15), batch_size=args.batch_size, length=args.length, updates=5)
    config = LMTrainingConfig(precision="fp32" if args.tiny else "bf16_mixed")
    seed_local(20260925, device)
    model = construct(case, args, device)
    optimizer, scheduler = optimizer_for(model, "compiled-native")
    tokenizer = None if args.tiny else load_native_tokenizer(args.artifacts)
    counts_by_rank = gather(model.counts(fixed_batch(case, tokenizer, 0, rank, tiny=args.tiny)))
    expected_active = sorted(active_names(model, case.mode()))
    configuration = graph_configuration(model, case, config, args, counts_by_rank, expected_active)
    configuration["source_checkpoint_sha256"] = report["source_checkpoint"]["sha256"]
    report.update(case=asdict(case), configuration=configuration)
    fingerprint = {"checkpoint_sha256": report["source_checkpoint"]["sha256"],
                   "source_hashes": report["sources"], "protocol_sha256": sha256_file(PROTOCOL)}
    counters = TrainingCounters()
    cursor = graph_cursor(case, rank, 0)
    generators = {"data": torch.Generator().manual_seed(7300 + rank),
                  "local": torch.Generator(device=device).manual_seed(7400 + rank)}
    seed_local(7500 + rank, device)
    step_calls = 0

    def count_step(*unused):
        nonlocal step_calls
        step_calls += 1
        report["physical_updates_per_rank"] = step_calls

    hook = optimizer.register_step_post_hook(count_step)

    def prepare(model, cursor, label):
        rng = tree_digests(_local_rng(device, generators))
        batch = move_batch(fixed_batch(case, tokenizer, cursor["next_update"], rank, tiny=args.tiny), device)
        adapter = PreparedDDPObjective(model, batch, mode=case.mode(),
            global_counts=configuration["global_counts"], world_size=2, config=config)
        runtime = DDPGraphTraining(adapter, expected_active_names=expected_active,
                                   gradient_as_bucket_view=False, bucket_cap_mb=25)
        coordinated("record graph preparation", lambda: persist(label + "/prepare"))
        runtime.prepare(warmup=11)
        publish(label + "/warmup_rng_exact", rng == tree_digests(_local_rng(device, generators)))
        return runtime

    def capture(runtime, label):
        rng = tree_digests(_local_rng(device, generators))
        coordinated("record graph capture", lambda: persist(label + "/capture"))
        runtime.capture(warmup=11, release_transient_cache=True)
        publish(label + "/capture_rng_exact", rng == tree_digests(_local_rng(device, generators)))

    def update(runtime, optimizer, scheduler, counters, cursor, label, *, replay):
        coordinated("graph update cursor", lambda: validate_graph_cursor(cursor, case, rank, counters))
        batch = fixed_batch(case, tokenizer, cursor["next_update"], rank, tiny=args.tiny)
        runtime.load_batch(batch)
        result = runtime.backward(replay=replay)
        raw = tree_digests({name: parameter.grad for name, parameter in runtime.model.named_parameters()
                           if parameter.grad is not None})
        replicas = gather(raw)
        publish(label + "/raw_gradient_replicas_exact", replicas[0] == replicas[1])
        raw_losses = tree_digests(result)
        metrics = finish_step(runtime, optimizer, scheduler, counters, result)
        next_cursor = graph_cursor(case, rank, counters.optimizer_updates)
        state = local_boundary(runtime.model, optimizer, scheduler, counters, next_cursor, device, generators)
        replicas = gather(state["state"])
        publish(label + "/state_replicas_exact", replicas[0] == replicas[1])
        publish(label + "/finite_state", state_health(runtime.model, optimizer)["passed"])
        record = {"batch": tree_digests(vars(batch)), "raw_losses": raw_losses,
                  "raw_gradients": raw, "metrics": metrics, "boundary": state, "replay": replay}
        records = gather(record)
        def record_update():
            if rank == 0:
                report.setdefault("updates", []).append({"label": label, "ranks": records})
                persist(label)
                tracker.log({"physical_updates_per_rank": step_calls, label + "/objective": metrics["objective"],
                             label + "/gradient_norm": metrics["gradient_norm_before_clip"]})
        coordinated("record graph update", record_update)
        return record, next_cursor

    def release(runtime, label):
        metadata = gather(runtime.metadata)
        report.setdefault("graph_lifecycles", []).append({"label": label, "ranks": metadata})
        rng = tree_digests(_local_rng(device, generators))
        coordinated("release graph and old reducer hooks", lambda: release_graph_and_hooks(runtime))
        publish(label + "/release_rng_exact", rng == tree_digests(_local_rng(device, generators)))

    try:
        runtime = prepare(model, cursor, "initial")
        for index in range(3):
            _, cursor = update(runtime, optimizer, scheduler, counters, cursor,
                               f"eager_preparation_{index + 1}", replay=False)
        runtime.load_batch(fixed_batch(case, tokenizer, cursor["next_update"], rank, tiny=args.tiny))
        capture(runtime, "initial")
        _, cursor = update(runtime, optimizer, scheduler, counters, cursor, "graphed_preparation", replay=True)
        release(runtime, "initial")
        del runtime
        gc.collect()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        boundary = local_boundary(model, optimizer, scheduler, counters, cursor, device, generators)
        required = sum(parameter.numel() * parameter.element_size() * (3 if parameter.requires_grad else 1)
                       for parameter in model.parameters()) + 2 ** 30
        def disk_preflight():
            path = ensure_persistent_checkpoint_directory(args.checkpoint_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(path.parent).free
            if free < required:
                raise OSError(f"Checkpoint needs approximately {required} bytes; only {free} are free")
            return {"estimated_required_bytes": required, "free_bytes": free, "path": str(path)}
        report["checkpoint_disk_preflight"] = gather(coordinated("disk preflight", disk_preflight))
        coordinated("record save", lambda: persist("save_checkpoint"))
        receipt = save_distributed_checkpoint(args.checkpoint_dir, model, optimizer, scheduler=scheduler,
            counters=counters, data_cursor=cursor, configuration=configuration, source_fingerprint=fingerprint,
            generators=generators, device=device)
        report["checkpoint"] = receipt
        publish("save_preserves_cleared_boundary", boundary == local_boundary(
            model, optimizer, scheduler, counters, cursor, device, generators))
        coordinated("record committed checkpoint", lambda: persist("checkpoint_committed"))
        runtime = prepare(model, cursor, "reference")
        capture(runtime, "reference")
        publish("reference_recapture_preserves_boundary", boundary == local_boundary(
            model, optimizer, scheduler, counters, cursor, device, generators))
        expected_draws = draw_rng(device, generators)
        reference, cursor = update(runtime, optimizer, scheduler, counters, cursor, "reference", replay=True)
        release(runtime, "reference")
        del runtime
        hook.remove()
        hook = None
        model.zero_grad(set_to_none=True)
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        coordinated("record reconstruction", lambda: persist("reconstruct_load"))
        model = construct(case, args, device)
        optimizer, scheduler = optimizer_for(model, "compiled-native")
        resumed_configuration = graph_configuration(model, case, config, args, counts_by_rank,
                                                    sorted(active_names(model, case.mode())))
        resumed_configuration["source_checkpoint_sha256"] = report["source_checkpoint"]["sha256"]
        publish("reconstructed_execution_configuration_exact", resumed_configuration == configuration)
        resumed = load_distributed_checkpoint(args.checkpoint_dir, model, optimizer, scheduler=scheduler,
            configuration=resumed_configuration, source_fingerprint=fingerprint, generators=generators,
            expected_manifest_sha256=receipt["manifest_sha256"], device=device)
        counters, cursor = resumed["counters"], resumed["data_cursor"]
        coordinated("restored graph cursor", lambda: validate_graph_cursor(cursor, case, rank, counters))
        publish("restored_cleared_boundary_exact", boundary == local_boundary(
            model, optimizer, scheduler, counters, cursor, device, generators))
        runtime = prepare(model, cursor, "restored")
        capture(runtime, "restored")
        publish("restored_recapture_preserves_boundary", boundary == local_boundary(
            model, optimizer, scheduler, counters, cursor, device, generators))
        publish("next_rank_local_random_draws_exact", expected_draws == draw_rng(device, generators))
        hook = optimizer.register_step_post_hook(count_step)
        restored, cursor = update(runtime, optimizer, scheduler, counters, cursor, "restored", replay=True)
        publish("continuation_batch_exact", reference["batch"] == restored["batch"])
        publish("continuation_raw_losses_exact", reference["raw_losses"] == restored["raw_losses"])
        publish("continuation_raw_gradients_exact", reference["raw_gradients"] == restored["raw_gradients"])
        publish("continuation_metrics_exact", reference["metrics"] == restored["metrics"])
        publish("continuation_full_state_rng_cursor_exact", reference["boundary"] == restored["boundary"])
        release(runtime, "restored")
        del runtime
        gc.collect()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        publish("physical_update_accounting", step_calls == 6 and counters.optimizer_updates == 5)
        publish("warmup_capture_accounting", sum(item["ranks"][rank]["warmup_backward_calls"]
            for item in report["graph_lifecycles"]) == 33 and sum(item["ranks"][rank]["capture_backward_calls"]
            for item in report["graph_lifecycles"]) == 3)
        publish("checkpoint_retained", (args.checkpoint_dir / "manifest.json").is_file()
                and (args.checkpoint_dir / "state.pt").stat().st_size == receipt["state"]["size_bytes"])
        report.update(physical_updates_per_rank=step_calls, total_rank_optimizer_steps=sum(gather(step_calls)),
                      logical_endpoint_updates=counters.optimizer_updates,
                      checkpoint_disposal={"deleted": False, "policy": "Retain until separately verified GCS upload"})
    finally:
        if hook is not None:
            hook.remove()


def main(argv=None):
    args = parse_args(argv)
    if os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", os.environ.get("NCCL_ASYNC_ERROR_HANDLING")) != "0":
        raise RuntimeError("Launch graph recovery with TORCH_NCCL_ASYNC_ERROR_HANDLING=0 and an external timeout")
    configure_determinism(True)
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    runtime = require_container_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    compiler_configuration()
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
                            timeout=timedelta(minutes=30))
    if dist.get_world_size() != 2:
        raise RuntimeError("Graph recovery requires two ranks on two actual GPUs")
    rank = dist.get_rank()
    tracker = None
    original_error = None
    report = {"schema": "olmo-two-gpu-graph-recovery-v1", "started_utc": utc_now(), "runtime": runtime,
              "passed": False, "status": "running", "checks": [], "physical_updates_per_rank": 0,
              "scope": "same_world_size_checkpoint_and_both_continuations_rebuild_actual_ddp_graphs"}

    def persist(stage=None):
        if stage is not None:
            report["stage"] = stage
        if rank == 0:
            write_json(args.output_dir / "progress.json", report)

    def publish(name, passed):
        flags = gather(bool(passed))
        row = {"name": name, "passed": all(flags), "rank_passed": flags}
        report["checks"].append(row)
        coordinated("record graph recovery gate", persist)
        if not row["passed"]:
            raise AssertionError(f"Graph recovery gate failed: {name}; rank flags={flags}")

    try:
        def setup():
            nonlocal tracker
            if rank != 0:
                return None
            args.output_dir.mkdir(parents=True, exist_ok=False)
            sources = [*sorted((ROOT / "cdrm/pretrained").glob("*.py")),
                       *sorted((ROOT / "scripts").glob("olmo*.py")),
                       ROOT / "scripts/experiment_tracking.py", PROTOCOL]
            hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
            for path in sources:
                target = args.output_dir / "source-snapshot" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            hook_source = inspect.getsource(DistributedDataParallel._remove_autograd_hooks)
            checkpoint = ({"sha256": "0" * 64, "kind": "deterministic_tiny_fixture"} if args.tiny else
                          validate_prepared_manifest(args.artifacts)["checkpoint"])
            tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-two-gpu",
                name=args.output_dir.name, output_dir=args.output_dir, preserve_state=preserve_local_rng)
            tracker.start({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
            print(tracker.record["run_url"], flush=True)
            return {"sources": hashes, "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "dependencies": dependency_record(args.output_dir, include_dao=False, include_fa4=False),
                "source_checkpoint": checkpoint, "wandb": tracker.record,
                "ddp_teardown_api": {"name": "DistributedDataParallel._remove_autograd_hooks",
                    "torch_version": str(torch.__version__), "source": hook_source,
                    "source_sha256": hashlib.sha256(hook_source.encode()).hexdigest()}}
        report.update(gather(coordinated("rank-zero setup", setup))[0])
        if rank == 0:
            report["wandb"] = tracker.record
        coordinated("record setup", lambda: persist("setup"))
        with disable_autocast_weight_cache(), backend_context("math" if args.tiny else "flash"):
            run(args, report, tracker, persist, publish)
        publish("frozen_source_snapshot", all(sha256_file(ROOT / name) == digest and
            sha256_file(args.output_dir / "source-snapshot" / name) == digest for name, digest in report["sources"].items()))
        report.update(status="passed", passed=True, stage="complete")
    except BaseException as error:
        original_error = error
        report.update(status="failed", passed=False, error={"type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        report["finished_utc"] = utc_now()
        if rank == 0 and args.output_dir.is_dir():
            write_json(args.output_dir / "report.json", report)
            try:
                if tracker is not None:
                    finish_tracking(tracker, report, original_error=original_error)
            finally:
                report["passed"] = report["status"] == "passed"
                write_json(args.output_dir / "report.json", report)
                persist()
        if original_error is None:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
