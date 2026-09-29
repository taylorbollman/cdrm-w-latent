#!/usr/bin/env python3
"""Explicit three-update B-only execution of a pinned CPU-resolved draft.

This separate operator entrypoint grants no authority to the resolver. It is
bounded lifecycle acceptance, not the general campaign or quality launcher.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
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
    build_campaign_model, build_campaign_adamw, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.distributed_checkpoint import (DistributedCheckpointError,
    load_distributed_checkpoint, save_distributed_checkpoint)
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.packed_campaign_data import PackedCampaignData
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, validate_prepared_manifest
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts import olmo_campaign_manifest as manifest_resolver
from scripts.olmo_campaign_base_loop import (model_contract, validate_cursor,
    plan_updates, source_hashes as base_sources)
from scripts.olmo_campaign_loop import Coordinator, LifecycleError, LoopPolicy, StopRequest, run_loop
from scripts.olmo_campaign_loop_guarded import retain_without_rng, run_stage_releasing_failure
from scripts.olmo_campaign_loop_run import finalize_report, storage_location, validate_replica_rows
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_restart import boundary, checkpoint_disk_preflight
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import state_health
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_packed_campaign_run import configure_cuda_runtime, cursor_record
from scripts.olmo_two_gpu_recovery import seed_local
from scripts.olmo_two_gpu_validate import preserve_local_rng

SCHEMA = "olmo-campaign-manifest-run-v1"
WORLD_SIZE, WARMUP = 2, 11
PROTOCOL = ROOT / "docs/reports/olmo-campaign-manifest/run-protocol.md"


def source_hashes():
    sources = {**base_sources(), **manifest_resolver.source_hashes()}
    for path in (Path(__file__), ROOT/"tests/test_campaign_manifest_run.py", PROTOCOL):
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
    return dict(sorted(sources.items()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("reference", "resume"), required=True)
    for name in ("manifest", "resolved", "output-dir", "checkpoint-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--resolved-sha256", required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-manifest-sha256")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args(argv)
    fields = (args.resume, args.resume_manifest_sha256, args.reference_report, args.reference_sha256)
    if ((args.phase == "reference" and any(v is not None for v in fields))
            or (args.phase == "resume" and any(v is None for v in fields))):
        parser.error("Only resume takes a checkpoint and reference, both with independent SHA256 pins")
    for value in (args.manifest_sha256, args.resolved_sha256, args.resume_manifest_sha256, args.reference_sha256):
        if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            parser.error("Use independent lowercase SHA256 pins")
    args.output_dir, args.checkpoint_root = args.output_dir.resolve(), args.checkpoint_root.resolve()
    if not args.output_dir.is_relative_to(ROOT) or not args.checkpoint_root.is_relative_to(ROOT):
        parser.error("Keep evidence and full checkpoints under the persistent project checkout")
    return args


def validate_execution(manifest):
    """Reject every declaration outside this separately authorized adapter."""
    recipes = manifest_resolver.validate_manifest(manifest, manifest_resolver.source_hashes())
    expected = {**manifest_resolver.EXECUTION_COMMON, **manifest_resolver.PATHS['bf16_mixed'],
                'precision':'bf16_mixed', 'graph_mode':'prepared_cuda_graph'}
    retention = manifest['retention']
    if (list(recipes) != ['B'] or manifest['execution'] != expected
            or manifest['budget']['updates'] != 3
            or manifest['budget']['target_valid_tokens_per_update'] != 16384
            or manifest['partition'] != {'world_size':2, 'physical_batch_per_rank':{'B':8}}
            or manifest['evaluation']['kind'] != 'deferred'
            or retention['checkpoint_every_updates'] != 3
            or retention['keep_local_completed'] < 3):
        raise ValueError('Adapter accepts only B/T1024/two ranks/B8/BF16 captured/3x16384/deferred eval; save every3, retain>=3')
    storage_location(retention['storage_prefix'])
    return recipes['B']


def load_draft(args):
    """Re-resolve all local byte authorities before initializing CUDA."""
    manifest = manifest_resolver.read_json(args.manifest, args.manifest_sha256)
    resolved = manifest_resolver.read_json(args.resolved, args.resolved_sha256)
    recipe = validate_execution(manifest)
    actual = manifest_resolver.resolve(manifest)
    actual['manifest_file_sha256'] = args.manifest_sha256
    if actual != resolved:
        raise ValueError('Pinned resolved draft differs from independently re-resolved declaration')
    return manifest, resolved, recipe


def construct_declared(manifest, resolved, recipe, device):
    """Native constructor consumes the manifest, not legacy probe defaults."""
    artifacts = manifest_resolver.local_path(manifest['model']['artifacts'])
    authority = validate_prepared_manifest(artifacts)
    if authority['checkpoint'] != resolved['model_checkpoint_authority']:
        raise ValueError('Model checkpoint authority changed after CPU resolution')
    state = load_native_state_dict(artifacts)
    execution = manifest['execution']
    base = OLMoTiledRTForCausalLM(OLMoConfig.native_1b(), device='meta', dtype=torch.float32,
        attention_backend='sdpa', attention_precision=execution['rt_attention_precision'],
        ordinary_activation_checkpointing=execution['ordinary_activation_checkpointing'],
        cast_weights_once=execution['cast_weights_once'], tile_backend=execution['rt_forward_tiles'],
        backward_tile_backend=execution['rt_backward_tiles'], backward_memory=execution['backward_memory'],
        reuse_rope=execution['reuse_rope'], kv_only_writes=execution['kv_only_writes'],
        ordinary_pointwise_backend=execution['ordinary_pointwise_backend'],
        ordinary_rope_backend=execution['ordinary_rope_backend'])
    base.load_state_dict(state, strict=True, assign=True)
    del state
    model = build_campaign_model(base, recipe).to(device).train()
    if (model.backbone.backbone.config.to_dict() != resolved['native_backbone_config']
            or model.config.to_dict() != resolved['nextlat_configs']['B']
            or recipe.to_dict() != resolved['recipes']['B']):
        raise ValueError('Constructed configuration differs from resolved declaration')
    return model, authority['checkpoint']


def assert_declared_plan(data, manifest, resolved):
    actual = manifest_resolver.plan_updates(data, manifest['budget'],
        {'B':(manifest['partition']['world_size'], manifest['partition']['physical_batch_per_rank']['B'])})
    if actual != resolved['plan']:
        raise ValueError('Actual pure data plan differs from pinned resolution')
    return plan_updates(data, tuple(row['counts']['valid_tokens'] for row in actual['updates']))


def assert_declared_ownership(contract, resolved):
    inventory = resolved['resource_cards']['B']['parameters']['observed_inventory_from_retained_ledger']
    if (contract['resident_parameters'] != inventory['registered_unique']
            or contract['trainable_parameters'] != inventory['trainable']
            or contract['trainable_parameters'] != inventory['optimizer_owned']):
        raise ValueError('Actual ordinary ownership differs from resolved card')


def checkpoint_policy(manifest):
    retention = manifest['retention']
    # Extra checkpoint1 is the explicit restart diagnostic; final3 is the
    # declared cadence. Time-triggered checkpoints may additionally occur.
    return LoopPolicy(manifest['budget']['updates'], retention['checkpoint_seconds'],
        tuple(sorted({1, *range(retention['checkpoint_every_updates'],
            manifest['budget']['updates']+1, retention['checkpoint_every_updates'])})), save_initial=False)


def load_reference(path, sha, sources, configuration, manifest_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024*1024 or sha256_file(path) != sha:
        raise ValueError("Reference report differs from bounded SHA authority")
    result = json.loads(path.read_text())
    saved = [r for r in result.get("local_checkpoints", []) if r.get("optimizer_update") == 1]
    if (sha256_file(path) != sha or result.get("schema") != SCHEMA or result.get("phase") != "reference"
            or result.get("status") != "passed" or result.get("sources") != sources
            or result.get("configuration") != configuration or set(result.get("updates", {})) != {"1","2","3"}
            or result.get('completed_segment') is not True
            or len(saved) != 1 or saved[0]["receipt"].get("manifest_sha256") != manifest_sha256
            or len(saved[0].get("boundary_by_rank", [])) != WORLD_SIZE):
        raise ValueError("Require completed identical reference and its update1 checkpoint")
    published = [r for r in result.get("published_checkpoints", []) if r.get("manifest_sha256") == manifest_sha256]
    if len(published) != 1 or not published[0].get("retention", {}).get("download_sha256_verified"):
        raise ValueError("Reference update1 checkpoint must have verified retained objects")
    return result, saved[0]["boundary_by_rank"]


def assert_inputs(args):
    if (sha256_file(args.manifest) != args.manifest_sha256
            or sha256_file(args.resolved) != args.resolved_sha256):
        raise ValueError('Pinned manifest or resolved evidence changed during execution')
    if args.phase == 'resume' and sha256_file(args.reference_report) != args.reference_sha256:
        raise ValueError('Pinned reference report changed during execution')


def run_stage(args, coordinator, device, runtime, determinism, report, tracker):
    rank = coordinator.rank
    manifest, resolved, recipe = args.draft
    declarations = manifest['data']
    batch_size = manifest['partition']['physical_batch_per_rank']['B']
    length = recipe.sequence_length
    targets = tuple(row['counts']['valid_tokens'] for row in resolved['plan']['updates'])
    maximum = manifest['budget']['updates']
    prefix = manifest['retention']['storage_prefix']+'/'+args.phase
    corpus = manifest_resolver.local_path(declarations['corpus'])
    index_path = manifest_resolver.local_path(declarations['index'])
    data = coordinator.call("open packed corpus", lambda: PackedCampaignData(corpus, index_path))
    try:
        def check_data():
            if data.length != length or data.manifest_sha256 != declarations['index_manifest_sha256'] or data.split != "train":
                raise ValueError("Use the pinned original T1024 train index")
        coordinator.call("pinned data selection", check_data)
        plans = coordinator.call("finite diagnostic plan", lambda: assert_declared_plan(data, manifest, resolved))
        model, source = coordinator.call("pretrained ordinary B", lambda: construct_declared(manifest, resolved, recipe, device))
        optimizer = build_campaign_adamw(model, recipe, fused=True)
        scheduler = CampaignTokenSchedule(optimizer, targets, warmup_tokens=recipe.warmup_tokens,
                                         start_fraction=recipe.warmup_start_fraction)
        training = LMTrainingConfig(precision=manifest['execution']['precision'], max_grad_norm=recipe.max_grad_norm)
        contract = coordinator.call("ordinary ownership", lambda: model_contract(model, recipe, optimizer))
        coordinator.call("resolved ownership card", lambda: assert_declared_ownership(contract, resolved))
        if scheduler.plan_sha256 != resolved['schedule']['plan_sha256']:
            raise LifecycleError('Actual scheduler differs from resolved token prefix')
        dormant = tree_digests(model.backbone.fusion.state_dict())
        configuration = {"schema":SCHEMA, "recipe":recipe.to_dict(), "model":model.config.to_dict(),
            "backbone":model.backbone.backbone.config.to_dict(), "parameters":contract,
            "training":asdict(training), "targets":list(targets), "schedule":scheduler.checkpoint_contract(),
            "declarations":manifest, "resolved_sha256":args.resolved_sha256,
            "manifest_file_sha256":args.manifest_sha256, "resolved_plan_sha256":resolved['plan_sha256'],
            "operator_scope":"Separately invoked bounded B acceptance; CPU resolver grants no launch or numerical clearance",
            "world_size":WORLD_SIZE, "physical_batch_per_rank":batch_size, "context_length":length,
            "runtime":runtime, "determinism":determinism, "data_manifest":data.manifest,
            "data_manifest_sha256":data.manifest_sha256, "ddp":{"static_graph":True,
                "find_unused_parameters":False,"broadcast_buffers":False,"gradient_as_bucket_view":False,"bucket_cap_mb":25},
            "warmup":WARMUP, "seed_policy":"jitter_seed + rank for process and CPU generator; +world_size for CUDA generator", "dormant_fusion_state":dormant}
        # Canonical JSON types are also what a report supplies after reload.
        configuration = json.loads(json.dumps(configuration))
        fingerprint = {"checkpoint_sha256":source["sha256"], "checkpoint":source,
            "recipe_sha256":recipe.sha256,"manifest_sha256":args.manifest_sha256,"resolved_sha256":args.resolved_sha256,"data_manifest_sha256":data.manifest_sha256,"sources":report["sources"]}
        report.update(configuration=configuration, source_checkpoint=source, parameters=contract,
            logical_updates=[{"start_cursor":asdict(p.start_cursor),"next_cursor":asdict(p.next_cursor),
                             "counts":asdict(p.counts),"unique_documents":p.unique_document_count} for p in plans])
        reference, saved = (None, None) if args.phase == "reference" else coordinator.call("reference authority", lambda:
            load_reference(args.reference_report, args.reference_sha256, report["sources"], configuration, args.resume_manifest_sha256))
        generators = {"data":torch.Generator().manual_seed(recipe.jitter_seed+rank),
                      "local":torch.Generator(device=device).manual_seed(recipe.jitter_seed+WORLD_SIZE+rank)}
        seed_local(recipe.jitter_seed+rank, device)
        counters = TrainingCounters()
        def current_boundary():
            return boundary(model, optimizer, scheduler, counters,
                cursor_record(data.cursor(), rank=rank, batch_size=batch_size), device, generators)
        if args.phase == "resume":
            restored = load_distributed_checkpoint(args.resume, model, optimizer, scheduler=scheduler,
                configuration=configuration, source_fingerprint=fingerprint, generators=generators,
                expected_manifest_sha256=args.resume_manifest_sha256, device=device)
            counters = restored["counters"]
            if counters.optimizer_updates != 1:
                raise LifecycleError("Only the reference update1 boundary may be resumed")
            coordinator.call("restore committed cursor", lambda: validate_cursor(data, restored["data_cursor"], counters, plans, rank, batch_size=batch_size, restore=True))
            exact = coordinator.gather(current_boundary() == saved[rank])
            report["restored_boundary_exact"] = exact
            if not all(exact): raise LifecycleError("Restored complete model Adam RNG cursor differs")
            report["resume_checkpoint"] = restored["manifest"]
        def materialize(plan):
            packed = data.rank_batches(plan, rank=rank, world_size=WORLD_SIZE, physical_batch_size=batch_size)
            noises = tuple(feedback_noise_for_rows(recipe, keys, logical_update=plan.start_cursor.next_update,
                sequence_length=length,width=model.config.model_dim,physical_batch_size=batch_size) for keys in packed.keys)
            if any(noise is not None for noise in noises): raise ValueError("Ordinary model unexpectedly requested feedback jitter")
            return packed, noises
        def persist():
            report["wandb"] = tracker.record
            write_json(args.output_dir/"report.json", report)
        initial = current_boundary()
        packed, noises = coordinator.call("preparation data", lambda: materialize(plans[counters.optimizer_updates]))
        counts = sum_objective_counts(coordinator.gather(sum_objective_counts([model.counts(b) for b in packed.batches])))
        adapter = CampaignObjective(model, packed.batches[0], mode=recipe.mode(), global_counts=counts,
            world_size=WORLD_SIZE, feedback_noise=noises[0], config=training)
        runner = CampaignDDPGraphTraining(adapter)
        report["adam_resident_before_ddp"] = bool(optimizer.state)
        def phase(name, action):
            report["stage"] = "prepare/"+name+"/"+action
            coordinator.call("preparation evidence", persist, rank_zero=True)
        runner.prepare(warmup=WARMUP, phase_observer=phase)
        runner.capture(warmup=WARMUP, release_transient_cache=True, phase_observer=phase)
        report["preparation_boundary_exact"] = coordinator.gather(current_boundary() == initial)
        report["memory_after_capture"] = coordinator.gather(memory())
        if not all(report["preparation_boundary_exact"]): raise LifecycleError("Preparation changed model Adam RNG or clocks")
        del packed, noises
        def save(number, reason):
            if number == 0: raise LifecycleError("Original pretrained authority replaces checkpoint0")
            destination = args.checkpoint_root/f"update-{number:06d}"
            coordinator.call("checkpoint disk preflight", lambda: checkpoint_disk_preflight(destination, model))
            before = current_boundary()
            state = coordinator.gather(before["state"])
            if any(value != state[0] for value in state): raise LifecycleError("Checkpoint replicas differ")
            try:
                with runner.checkpoint_boundary():
                    receipt = save_distributed_checkpoint(destination, model, optimizer, scheduler=scheduler,
                        counters=counters, data_cursor=cursor_record(data.cursor(),rank=rank,batch_size=batch_size),
                        configuration=configuration,source_fingerprint=fingerprint,generators=generators,device=device)
            except DistributedCheckpointError:
                raise LifecycleError("Coordinated checkpoint failure; prior retained checkpoint is authoritative") from None
            if not all(coordinator.gather(before == current_boundary())):
                raise LifecycleError("Saving checkpoint changed live graph boundary")
            report.setdefault("local_checkpoints", []).append({"optimizer_update":number,"reason":reason,
                "receipt":receipt,"boundary_by_rank":coordinator.gather(before)})
            coordinator.call("checkpoint local evidence", persist, rank_zero=True)
            return receipt
        def publish(receipt):
            write_json(args.output_dir/"latest-checkpoint.json", receipt)
            report.setdefault("published_checkpoints", []).append(receipt)
            persist()
        def update():
            index = counters.optimizer_updates
            plan = plans[index]
            coordinator.call("validate committed cursor", lambda: validate_cursor(data,
                cursor_record(data.cursor(),rank=rank,batch_size=batch_size),counters,plans,rank,batch_size=batch_size))
            packed, noises = coordinator.call("actual update data", lambda: materialize(plan))
            inputs = tree_digests({"batches":[vars(b) for b in packed.batches],"noise":noises,
                                  "keys":packed.keys,"cursor":asdict(data.cursor())})
            torch.cuda.synchronize(); begin = time.perf_counter()
            result = runner.backward(packed.batches, feedback_noises=noises, replay=True)
            gradients = tree_digests({n:p.grad for n,p in model.named_parameters() if p.requires_grad})
            metrics = runner.step(result,optimizer,scheduler=scheduler,counters=counters)
            coordinator.call("commit completed cursor", lambda: data.commit(plan.start_cursor,plan))
            torch.cuda.synchronize(); elapsed = time.perf_counter()-begin
            row = {"input":inputs,"raw_gradients":gradients,"metrics":metrics,"boundary":current_boundary()}
            rows = coordinator.gather(row)
            report.setdefault("updates", {})[str(counters.optimizer_updates)] = rows
            report.setdefault("observations", {})[str(counters.optimizer_updates)] = {
                "rank_seconds_including_gradient_hash":coordinator.gather(elapsed),
                "memory_by_rank":coordinator.gather(memory()),"rank_data":coordinator.gather(packed.accounting)}
            coordinator.call("completed update evidence", persist, rank_zero=True)
            validate_replica_rows(rows)
            coordinator.call("post-update data counters", lambda: validate_cursor(data,
                cursor_record(data.cursor(),rank=rank,batch_size=batch_size),counters,plans,rank,batch_size=batch_size))
            health = coordinator.call("active and dormant state", lambda: {
                "finite":state_health(model,optimizer)["passed"],
                "dormant_exact":tree_digests(model.backbone.fusion.state_dict()) == dormant,
                "ownership_exact":model_contract(model,recipe,optimizer) == contract})
            if not all(all(value.values()) for value in coordinator.gather(health)):
                raise LifecycleError("Ordinary state health or ownership failed")
            if reference is not None and rows != reference["updates"][str(counters.optimizer_updates)]:
                raise LifecycleError("Fresh captured continuation differs bitwise from uninterrupted reference")
            return metrics
        def log(metrics):
            persist()
            tracker.log({"update":counters.optimizer_updates,**scalar_metrics(metrics,"train")},step=counters.optimizer_updates)
        policy = checkpoint_policy(manifest)
        stop = StopRequest(args.stop_file)
        with stop.installed():
            # Original weights + empty Adam + seeds are a retained/reconstructible
            # origin boundary. Never write a redundant checkpoint0 on early stop.
            report["loop"] = run_loop(coordinator=coordinator,policy=policy,completed=lambda:counters.optimizer_updates,
                update=update,log=log,save=save,publish_checkpoint=publish,stop=stop,restored=True,
                retain=lambda receipt:retain_without_rng(receipt,prefix))
        report["final_boundary_by_rank"] = coordinator.gather(current_boundary())
        report["runner_by_rank"] = coordinator.gather(runner.metadata)
        report["final_counters"] = asdict(counters)
        report["reference_comparison"] = "bitwise_equal_all_executed_updates" if reference is not None else "not_requested"
        report["completed_segment"] = counters.optimizer_updates == maximum
        coordinator.call("final stage evidence", persist, rank_zero=True)
    finally:
        data.close()


def main(argv=None):
    args = parse_args(argv)
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Use the project container")
    if os.environ.get("WORLD_SIZE") != "2" or any(os.environ.get(k) != "0" for k in
            ("NCCL_ASYNC_ERROR_HANDLING","TORCH_NCCL_ASYNC_ERROR_HANDLING")):
        raise RuntimeError("Use two ranks, both NCCL async flags0 and an external1800-second timeout")
    # Resolve before configure_cuda_runtime: its deterministic controls must
    # precede every CUDA initialization. Early host failure uses torchrun teardown.
    args.draft = load_draft(args)
    manifest, resolved, recipe = args.draft
    targets = tuple(row['counts']['valid_tokens'] for row in resolved['plan']['updates'])
    rank = int(os.environ["LOCAL_RANK"])
    device,runtime,determinism = configure_cuda_runtime(rank)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl",timeout=timedelta(seconds=600),device_id=device)
    coordinator,tracker = Coordinator(),None
    start = time.monotonic()
    report = {"schema":SCHEMA,"phase":args.phase,"status":"running","sources":coordinator.call("sources",source_hashes),
        "updates":{},"scope":"Ordinary pretrained BF16 packed host-loop integration; no model-quality, throughput or NFR numerical clearance",
        "targets":list(targets),"manifest_sha256":args.manifest_sha256,"resolved_sha256":args.resolved_sha256,"origin_checkpoint0":"Existing original pretrained authority plus explicit fresh optimizer/seed recipe",
        "timing_scope":"Includes diagnostic hashes/health/collectives and retention where stated; not optimized throughput"}
    def setup():
        nonlocal tracker
        args.output_dir.mkdir(parents=True,exist_ok=False)
        for name in report["sources"]:
            destination=args.output_dir/"source-snapshot"/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
        for source,name,expected in ((args.manifest,'input-manifest.json',args.manifest_sha256),
                                     (args.resolved,'resolved-plan.json',args.resolved_sha256)):
            shutil.copy2(source,args.output_dir/name)
            if sha256_file(args.output_dir/name) != expected:
                raise ValueError('Input evidence copy differs from frozen authority')
        tracker=OnlineTracker(project=manifest['tracking']['project'],entity=manifest['tracking']['entity'],
            output_dir=args.output_dir,group=manifest['tracking']['group'],
            name=args.output_dir.name,preserve_state=preserve_local_rng)
        tracker.start({"phase":args.phase,"targets":targets,"scope":report["scope"]})
        print({"wandb":tracker.record["run_url"]},flush=True)
    healthy,error = True,None
    try:
        coordinator.call("setup and tracking",setup,rank_zero=True)
        with disable_autocast_weight_cache(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            run_stage_releasing_failure(args,coordinator,device,runtime,determinism,report,tracker,stage=run_stage)
        coordinator.call("final source check",lambda:assert_sources(report["sources"]))
        coordinator.call('final manifest/resolved check', lambda: assert_inputs(args))
        report["status"] = "passed" if report["completed_segment"] else "stopped_at_boundary"
    except LifecycleError as exc:
        error=exc;report.update(status="failed",error={"type":type(exc).__name__,"message":str(exc)},failure_scope="coordinated ordinary failure")
    except BaseException as exc:
        healthy=False;error=exc;report.update(status="failed",error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()},
                                            failure_scope="unknown update/CUDA/NCCL; external teardown")
    finally:
        report["elapsed_seconds"]=time.monotonic()-start
        if healthy:
            try: finalize_report(args.output_dir,coordinator,report,tracker,succeeded=error is None)
            finally:
                gc.collect();dist.destroy_process_group()
        elif args.output_dir.is_dir():
            write_json(args.output_dir/f"rank-{rank}-failure.json",report)
        if error is not None: raise error


def assert_sources(expected):
    if source_hashes() != expected: raise ValueError("Frozen manifest-run sources changed")


if __name__ == "__main__": main()
