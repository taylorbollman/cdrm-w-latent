#!/usr/bin/env python3
"""Opt-in tiny two-GPU captured campaign lifecycle acceptance on real packed data.

This is a bounded integration of the reusable host loop, not pretrained training.
Three logical updates use M1/M2/M3 accumulation at T16/B2 per rank. Run independent
reference, stop-file, fresh-resume and coordinated logging-failure stages. Only
fresh processes load checkpoints, before DDP/capture. External timeout required.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
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
from cdrm.pretrained.campaign_recipe import (CampaignRecipe, CampaignTokenSchedule,
    build_campaign_adamw, build_campaign_model, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.distributed_checkpoint import (load_distributed_checkpoint,
    save_distributed_checkpoint, DistributedCheckpointError, MANIFEST_FILENAME, STATE_FILENAME)
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.packed_campaign_data import PackedCampaignData
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_loop import Coordinator, LifecycleError, LoopPolicy, StopRequest, run_loop
from scripts.olmo_campaign_restart import boundary, checkpoint_disk_preflight
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_f1_common import state_health
from scripts.olmo_packed_campaign_run import configure_cuda_runtime, source_hashes as packed_sources
from scripts.olmo_two_gpu_recovery import seed_local
from scripts.olmo_two_gpu_validate import preserve_local_rng

SCHEMA = "olmo-campaign-loop-acceptance-v1"
POLICY = "continuous-stream-v1"
TARGETS = (64, 128, 192)
PROTOCOL = ROOT / "docs/reports/olmo-campaign-lifecycle/loop-protocol.md"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("B", "NFR"), default="NFR")
    parser.add_argument("--max-updates", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--checkpoint-seconds", type=float, default=600)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--request-stop-after", type=int, choices=(1, 2))
    parser.add_argument("--inject-log-error-at", type=int, choices=(1, 2, 3))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-manifest-sha256")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--storage-prefix")
    args = parser.parse_args(argv)
    for pair in ((args.resume, args.resume_manifest_sha256),
                 (args.reference_report, args.reference_sha256)):
        if (pair[0] is None) != (pair[1] is None):
            parser.error("Checkpoint and reference paths require their SHA256 pins")
    if args.request_stop_after is not None and args.stop_file is None:
        parser.error("request-stop-after requires an explicit stop-file")
    if args.request_stop_after is not None and args.inject_log_error_at is not None:
        parser.error("Use separate stop and logging-failure stages")
    for value in (args.index_sha256, args.resume_manifest_sha256, args.reference_sha256):
        if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            parser.error("SHA256 pins must be lowercase hexadecimal")
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    if not args.output_dir.is_relative_to(ROOT.resolve()):
        parser.error("Small evidence must stay under the persistent project checkout")
    if not args.checkpoint_root.is_relative_to(Path("/home/taylorbollman")) and not args.checkpoint_root.is_relative_to(ROOT.resolve()):
        parser.error("Acceptance checkpoints require persistent project/home storage")
    try:
        LoopPolicy(args.max_updates, args.checkpoint_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.storage_prefix is not None:
        storage_location(args.storage_prefix)
    return args


def storage_location(prefix):
    root = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/"
    if not prefix.startswith(root) or not prefix[len(root):] or any(p in ("", ".", "..") for p in prefix[len(root):].split("/")):
        raise ValueError("Use a distinct immutable fusion-startup GCS stage prefix")
    return prefix[5:].split("/", 1)


def source_hashes():
    sources = packed_sources()
    names = [Path(__file__), ROOT/"scripts/olmo_campaign_loop.py", ROOT/"scripts/olmo_campaign_lifecycle.py", PROTOCOL,
             ROOT/"scripts/olmo_two_gpu_retain.py", ROOT/"scripts/openelm_retain.py",
             ROOT/"tests/test_campaign_loop.py", ROOT/"tests/test_campaign_loop_run.py"]
    sources.update({str(path.relative_to(ROOT)): sha256_file(path) for path in names})
    return dict(sorted(sources.items()))


def plan_updates(data):
    """Pure bounded metadata plan; no scan/cycling of the remaining corpus."""
    cursor, updates = data.cursor(), []
    if cursor.next_update:
        raise ValueError("Plan requires a fresh reader")
    for target in TARGETS:
        update = data.peek_update(cursor, target)
        if update is None or update.counts.valid_tokens != target:
            raise ValueError("Acceptance corpus needs at least 384 initial valid tokens")
        updates.append(update)
        cursor = update.next_cursor
    return tuple(updates)


def construct_tiny(data, arm, device):
    torch.manual_seed(20260929)
    config = replace(OLMoConfig.tiny(), vocab_size=data.manifest["vocab_size"],
        tokenizer_vocab_size=data.manifest["vocab_size"], eos_token_id=data.manifest["eos_id"],
        pad_token_id=data.pad_id)
    recipe = CampaignRecipe(arm, sequence_length=16, rt_layers=(0, 1),
        effective_valid_tokens=128, warmup_tokens=sum(TARGETS), document_policy=POLICY)
    base = OLMoTiledRTForCausalLM(config, attention_backend="math", attention_precision="fp32",
        tile_backend="eager", backward_tile_backend="eager", backward_memory="recompute",
        reuse_rope=True, kv_only_writes=True)
    model = build_campaign_model(base, recipe).to(device).train()
    return model, recipe


def cursor_record(data, rank):
    return {"schema": SCHEMA, "cursor": asdict(data.cursor()), "rank": rank,
            "world_size": 2, "physical_batch_per_rank": 2}


def validate_cursor(data, record, counters, plans, rank, *, restore=False):
    index = counters.optimizer_updates
    if type(index) is not int or not 0 <= index <= len(plans):
        raise ValueError("Invalid optimizer update counter")
    expected_cursor = plans[0].start_cursor if index == 0 else plans[index-1].next_cursor
    expected = {"schema": SCHEMA, "cursor": asdict(expected_cursor), "rank": rank,
                "world_size": 2, "physical_batch_per_rank": 2}
    expected_counts = {"input_tokens": sum(p.counts.valid_tokens for p in plans[:index]),
                       "documents": sum(p.counts.packed_rows for p in plans[:index]),
                       "microbatches": sum(2*((len(p.rows)+3)//4) for p in plans[:index])}
    if record != expected or any(getattr(counters, key) != value for key, value in expected_counts.items()):
        raise ValueError("Checkpoint cursor, optimizer and data accounting differ")
    if restore:
        data.restore_cursor(record["cursor"])
    if data.cursor() != expected_cursor:
        raise ValueError("Reader differs from committed cursor")
    return expected_cursor


def validate_replica_rows(rows):
    if len(rows) != 2:
        raise LifecycleError("Exactly two replica observations required")
    for key in ("raw_gradients", "metrics"):
        if rows[0][key] != rows[1][key]:
            raise LifecycleError(f"Post-update {key} replicas differ")
    if rows[0]["boundary"]["state"] != rows[1]["boundary"]["state"]:
        raise LifecycleError("Post-update model/Adam/counter replicas differ")
    # Rank-local RNG, inputs and cursor rank ownership intentionally differ.


def load_reference(path, digest, sources, configuration):
    if sha256_file(path) != digest:
        raise ValueError("Reference report SHA256 differs")
    reference = json.loads(path.read_text())
    if (reference.get("schema") != SCHEMA or reference.get("status") != "passed"
            or reference.get("sources") != sources or reference.get("configuration") != configuration
            or len(reference.get("updates", {})) != 3):
        raise ValueError("Reference must be a complete matching three-update acceptance")
    return reference


def retain_checkpoint(receipt, prefix):
    from google.cloud import storage
    from scripts.openelm_retain import file_digest
    from scripts.olmo_two_gpu_retain import upload_verified
    bucket_name, key = storage_location(prefix)
    bucket = storage.Client().bucket(bucket_name)
    directory = Path(receipt["directory"])
    records = []
    # Manifest is last: its remote presence is the publication marker.
    for name in (STATE_FILENAME, MANIFEST_FILENAME):
        path = directory/name
        records.append(upload_verified(bucket, f"{key}/{directory.name}/{name}", path,
                                      file_digest(path), download_sha256=True))
    return {"objects": records, "create_only": True, "download_sha256_verified": True}


def run_stage(args, coordinator, device, runtime, determinism, report, tracker):
    rank = coordinator.rank
    data = coordinator.call("open packed data", lambda: PackedCampaignData(args.corpus, args.index))
    try:
        def validate_data():
            if data.length != 16 or data.manifest_sha256 != args.index_sha256:
                raise ValueError("Use the explicitly pinned T16 packed index")
        coordinator.call("data selection", validate_data)
        plans = coordinator.call("finite plan", lambda: plan_updates(data))
        model, recipe = coordinator.call("tiny model", lambda: construct_tiny(data, args.arm, device))
        optimizer = build_campaign_adamw(model, recipe, fused=True)
        scheduler = CampaignTokenSchedule(optimizer, TARGETS, warmup_tokens=recipe.warmup_tokens,
                                         start_fraction=recipe.warmup_start_fraction)
        training = LMTrainingConfig(precision="fp32", max_grad_norm=recipe.max_grad_norm)
        configuration = {"schema": SCHEMA, "scale": "tiny_native_vocabulary", "recipe": recipe.to_dict(),
            "resolved_precision": "fp32_math_eager_native_rt", "model": model.config.to_dict(),
            "backbone": model.backbone.backbone.config.to_dict(),
            "data_manifest_sha256": data.manifest_sha256, "training": asdict(training),
            "targets": list(TARGETS), "schedule": scheduler.checkpoint_contract(), "world_size": 2,
            "physical_batch_per_rank": 2, "runtime": runtime, "determinism": determinism,
            "ddp": {"static_graph": True, "broadcast_buffers": False, "gradient_as_bucket_view": False,
                    "bucket_cap_mb": 25}, "warmup": 11}
        fingerprint = {"initialization": "seeded_tiny_native_vocabulary_20260929",
            "initial_state": tree_digests(model.state_dict()), "recipe_sha256": recipe.sha256,
            "data_manifest_sha256": data.manifest_sha256, "sources": report["sources"]}
        fingerprint["sha256"] = hashlib.sha256(json.dumps(fingerprint["initial_state"],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report.update(configuration=configuration, fingerprint=fingerprint)
        reference = None if args.reference_report is None else coordinator.call("reference", lambda:
            load_reference(args.reference_report, args.reference_sha256, report["sources"], configuration))
        generators = {"data": torch.Generator().manual_seed(8300+rank),
                      "local": torch.Generator(device=device).manual_seed(8400+rank)}
        seed_local(8500+rank, device)
        counters = TrainingCounters()
        if args.resume is not None:
            loaded = load_distributed_checkpoint(args.resume, model, optimizer, scheduler=scheduler,
                configuration=configuration, source_fingerprint=fingerprint, generators=generators,
                expected_manifest_sha256=args.resume_manifest_sha256, device=device)
            counters = loaded["counters"]
            coordinator.call("restore cursor", lambda: validate_cursor(
                data, loaded["data_cursor"], counters, plans, rank, restore=True))
            report["resumed_checkpoint"] = loaded["manifest"]
        if counters.optimizer_updates >= args.max_updates:
            raise ValueError("This acceptance stage must perform at least one update")

        def current_boundary():
            return boundary(model, optimizer, scheduler, counters, cursor_record(data, rank), device, generators)

        def materialize(plan):
            packed = data.rank_batches(plan, rank=rank, world_size=2, physical_batch_size=2)
            noises = tuple(feedback_noise_for_rows(recipe, keys, logical_update=plan.start_cursor.next_update,
                sequence_length=16, width=model.config.model_dim, physical_batch_size=2) for keys in packed.keys)
            return packed, noises

        initial = current_boundary()
        packed, noises = coordinator.call("preparation data", lambda: materialize(plans[counters.optimizer_updates]))
        counts = sum_objective_counts(coordinator.gather(sum_objective_counts([model.counts(b) for b in packed.batches])))
        adapter = CampaignObjective(model, packed.batches[0], mode=recipe.mode(), global_counts=counts,
            world_size=2, feedback_noise=noises[0], config=training)
        runner = CampaignDDPGraphTraining(adapter)
        runner.prepare(warmup=11)
        runner.capture(warmup=11, release_transient_cache=True)
        preserved = coordinator.gather(current_boundary() == initial)
        if not all(preserved):
            raise LifecycleError("Graph preparation changed restored state")
        report["preparation_boundary_exact"] = preserved
        del packed, noises

        def persist():
            report["wandb"] = tracker.record
            write_json(args.output_dir/"report.json", report)

        def save(update_number, reason):
            destination = args.checkpoint_root/f"update-{update_number:06d}"
            coordinator.call("checkpoint disk", lambda: checkpoint_disk_preflight(destination, model))
            saved = current_boundary()
            replicas = coordinator.gather(saved["state"])
            if any(value != replicas[0] for value in replicas[1:]):
                raise LifecycleError("Checkpoint model/Adam/counter replicas differ")
            try:
                with runner.checkpoint_boundary():
                    receipt = save_distributed_checkpoint(destination, model, optimizer, scheduler=scheduler,
                        counters=counters, data_cursor=cursor_record(data, rank), configuration=configuration,
                        source_fingerprint=fingerprint, generators=generators, device=device)
            except DistributedCheckpointError:
                raise LifecycleError("Coordinated checkpoint save failed; prior checkpoints retained") from None
            same = coordinator.gather(saved == current_boundary())
            if not all(same):
                raise LifecycleError("Checkpoint changed live captured state")
            report.setdefault("local_checkpoints", []).append({"reason": reason, "receipt": receipt,
                                                              "boundary_by_rank": coordinator.gather(saved)})
            coordinator.call("local checkpoint record", persist, rank_zero=True)
            return receipt

        def publish_checkpoint(receipt):
            # Mutable local discovery pointer; every checkpoint and cloud object
            # it points to was committed immutably before this publication.
            write_json(args.output_dir/"latest-checkpoint.json", receipt)
            report.setdefault("published_checkpoints", []).append(receipt)
            persist()

        def update():
            index = counters.optimizer_updates
            plan = plans[index]
            coordinator.call("committed cursor", lambda: validate_cursor(
                data, cursor_record(data, rank), counters, plans, rank))
            packed, noises = coordinator.call("update data", lambda: materialize(plan))
            inputs = tree_digests({"batches": [vars(batch) for batch in packed.batches], "noise": noises,
                                  "keys": packed.keys, "cursor": asdict(data.cursor())})
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = runner.backward(packed.batches, feedback_noises=noises, replay=True)
            gradients = tree_digests({name: parameter.grad for name, parameter in model.named_parameters()
                                     if parameter.grad is not None})
            metrics = runner.step(result, optimizer, scheduler=scheduler, counters=counters)
            coordinator.call("commit cursor", lambda: data.commit(plan.start_cursor, plan))
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            local = {"input": inputs, "raw_gradients": gradients, "metrics": metrics,
                     "boundary": current_boundary()}
            rows = coordinator.gather(local)
            validate_replica_rows(rows)
            health = coordinator.call("state health", lambda: state_health(model, optimizer)["passed"])
            if not all(coordinator.gather(health)):
                raise LifecycleError("An updated model or Adam state is nonfinite")
            if reference is not None and rows != reference["updates"][str(counters.optimizer_updates)]:
                raise LifecycleError("Fresh continuation differs bitwise from uninterrupted reference")
            report.setdefault("updates", {})[str(counters.optimizer_updates)] = rows
            report.setdefault("update_timing_seconds", {})[str(counters.optimizer_updates)] = coordinator.gather(seconds)
            return metrics

        def log(metrics):
            update_number = counters.optimizer_updates
            # Local report is persisted before W&B; failure leaves inspectable
            # completed-update evidence on disk on the injected failure path.
            persist()
            if args.inject_log_error_at == update_number:
                raise OSError("deliberate rank-zero logging failure")
            tracker.log({"update": update_number, **scalar_metrics(metrics, "train")}, step=update_number)
            if args.request_stop_after == update_number:
                args.stop_file.parent.mkdir(parents=True, exist_ok=True)
                with args.stop_file.open("x") as stream:
                    stream.write("bounded acceptance stop request\n")

        stop = StopRequest(args.stop_file)
        policy = LoopPolicy(args.max_updates, args.checkpoint_seconds, (1, 2, 3))
        with stop.installed():
            report["loop"] = run_loop(coordinator=coordinator, policy=policy,
                completed=lambda: counters.optimizer_updates, update=update, log=log, save=save,
                publish_checkpoint=publish_checkpoint, stop=stop, restored=args.resume is not None,
                retain=None if args.storage_prefix is None else lambda receipt: retain_checkpoint(receipt, args.storage_prefix))
        report["runner_by_rank"] = coordinator.gather(runner.metadata)
        report["final_boundary_by_rank"] = coordinator.gather(current_boundary())
        report["reference_comparison"] = "bitwise_equal_all_executed_updates" if reference is not None else "not_requested"
        coordinator.call("final stage report", persist, rank_zero=True)
    finally:
        data.close()


def finalize_report(output_dir, coordinator, report, tracker, *, succeeded):
    def persist():
        if Path(output_dir).is_dir():
            if tracker is not None:
                report["wandb"] = tracker.record
            write_json(Path(output_dir)/"report.json", report)

    def finish():
        persist()
        if tracker is not None:
            tracker.finish(succeeded=succeeded)
            persist()
    try:
        coordinator.call("final report and tracking", finish, rank_zero=True)
    except LifecycleError:
        report.update(status="failed", finalization_error="coordinated final report or tracking failure")
        # If report storage itself failed this may also fail, consistently on
        # every rank. No additional model update or checkpoint is attempted.
        coordinator.call("record finalization failure", persist, rank_zero=True)
        raise


def main(argv=None):
    args = parse_args(argv)
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Use the project Docker GPU launcher")
    if os.environ.get("WORLD_SIZE") != "2":
        raise RuntimeError("Exactly two torchrun ranks required")
    if any(os.environ.get(key) != "0" for key in ("NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING")):
        raise RuntimeError("Set both NCCL async error flags=0 and use an external timeout")
    rank = int(os.environ["LOCAL_RANK"])
    device, runtime, determinism = configure_cuda_runtime(rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    coordinator, tracker = Coordinator(), None
    started = time.monotonic()
    report = {"schema": SCHEMA, "status": "running", "sources": coordinator.call("source inventory", source_hashes), "updates": {},
        "scope": "Tiny FP32 T16 captured-DDP lifecycle only; real corpus token IDs; no pretrained, BF16, throughput or H200 clearance.",
        "segment": {"max_updates": args.max_updates, "stop_file": None if args.stop_file is None else str(args.stop_file),
                    "request_stop_after": args.request_stop_after, "inject_log_error_at": args.inject_log_error_at},
        "timing_scope": "Includes diagnostic hashes and collectives; not a throughput benchmark."}

    def setup():
        nonlocal tracker
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name in report["sources"]:
            destination = args.output_dir/"source-snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, destination)
        tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
            group="olmo-campaign-loop", name=args.output_dir.name, preserve_state=preserve_local_rng)
        tracker.start({"arm": args.arm, "scope": report["scope"], **report["segment"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)

    healthy = True
    error = None
    try:
        coordinator.call("setup and online tracking", setup, rank_zero=True)
        with disable_autocast_weight_cache(), sdpa_kernel(SDPBackend.MATH):
            run_stage(args, coordinator, device, runtime, determinism, report, tracker)
        coordinator.call("final source integrity", lambda: assert_sources(report["sources"]))
        report["status"] = "passed"
    except LifecycleError as exc:
        error = exc
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      failure_scope="coordinated ordinary host failure")
    except BaseException as exc:
        # Do not launch more collectives from an unknown NCCL/update failure.
        healthy = False
        error = exc
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, failure_scope="unclassified; external launcher tears down peers")
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        if healthy:
            try:
                finalize_report(args.output_dir, coordinator, report, tracker, succeeded=error is None)
            finally:
                dist.destroy_process_group()
        else:
            # Best-effort local rank evidence, never a checkpoint of uncertain state.
            if args.output_dir.is_dir():
                write_json(args.output_dir/f"rank-{rank}-failure.json", report)
        if error is not None:
            raise error


def assert_sources(expected):
    if source_hashes() != expected:
        raise ValueError("Pinned runtime sources changed")


if __name__ == "__main__":
    main()
