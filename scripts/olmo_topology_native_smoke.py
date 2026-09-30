#!/usr/bin/env python3
"""Bounded native N/R/NR/FR integration through the current packed DDP graphs.

Launch in the project GPU container with two-rank torchrun and an external
whole-job timeout. Two B12 slots per rank give 48 real T1024 rows/update; two
updates exercise changed inputs. This is functionality evidence, not a B512
training comparison, throughput benchmark, BF16 equivalence, or restart test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from cdrm.pretrained.campaign_recipe import (CampaignRecipe, CampaignTokenSchedule,
    build_campaign_model, build_campaign_adamw, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, optimizer_ownership
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import (load_native_state_dict, validate_prepared_manifest,
    MANIFEST_FILENAME, FILE_SPECS)
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_allocation_acceptance import literal_counts
from scripts.olmo_allocation_benchmark import release_completed_graph_runner, source_hashes as allocation_sources
from scripts.olmo_campaign_graph_probe import pointer_snapshot, gradients_are_zero
from scripts.olmo_campaign_loop import Coordinator
from scripts.olmo_campaign_probe import memory
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_lm_common import tensor_digest
from scripts.olmo_packed_campaign_run import configure_cuda_runtime
from scripts.olmo_pilot_ordered_data import OrderedCampaignData
from scripts.olmo_two_gpu_validate import preserve_local_rng

SCHEMA = "olmo-topology-native-smoke-v1"
INITIALIZATION_SEED = 20261001
ARMS = ("N", "R", "NR", "FR")
LENGTH, PHYSICAL_BATCH, WORLD_SIZE, SLOTS, UPDATES = 1024, 12, 2, 2, 2
REAL_ROWS = PHYSICAL_BATCH * WORLD_SIZE * SLOTS
VALID_TOKENS = REAL_ROWS * LENGTH
EXECUTION = {"precision": "bf16_mixed", "master_dtype": "float32", "optimizer_state_dtype": "float32",
    "ordinary_attention": "flash_sdpa", "rt_attention_precision": "mixed",
    "rt_forward_tiles": "triton", "rt_backward_tiles": "triton", "backward_memory": "recompute",
    "ordinary_activation_checkpointing": True, "cast_weights_once": True, "reuse_rope": True,
    "kv_only_writes": True, "ordinary_pointwise_backend": "eager", "ordinary_rope_backend": "native",
    "graph_mode": "prepared_cuda_graph", "optimizer": "adamw_fused", "tf32": False,
    "autocast_cache": False, "torch_compile": False, "distributed": "replicated_static_ddp"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes():
    result = allocation_sources()
    for name in ("olmo_topology_native_smoke.py", "olmo_allocation_acceptance.py",
                 "olmo_campaign_graph_probe.py", "olmo_campaign_loop.py", "olmo_campaign_lifecycle.py",
                 "olmo_lm_common.py"):
        result["scripts/" + name] = sha256_file(ROOT / "scripts" / name)
    return dict(sorted(result.items()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    args = parser.parse_args(argv)
    if len(args.index_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.index_sha256):
        parser.error("Require the independently supplied ordered-index SHA256")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == ROOT.resolve() or not args.output_dir.is_relative_to(ROOT.resolve()):
        parser.error("Small evidence must use a new subdirectory of the persistent project")
    # Existence is checked once by rank zero after rendezvous, avoiding a race
    # between torchrun workers parsing arguments and creating the directory.
    return args


def smoke_recipe(arm):
    if arm not in ARMS:
        raise ValueError("This bounded smoke covers N/R/NR/FR only")
    return CampaignRecipe(arm, sequence_length=LENGTH, effective_valid_tokens=VALID_TOKENS,
                          document_policy="continuous-stream-v1")


def apply_smoke_objective(model):
    """Current reduced KL branch, explicitly recorded instead of recipe default1."""
    model.config = replace(model.config, lambda_kl=0.1, lambda_latent=1.0)
    if model.predictor is not None:
        model.predictor.config = replace(model.predictor.config, lambda_kl=0.1, lambda_latent=1.0)



def construct_model(state, recipe, device, *, model_config=None):
    """Common rank initialization; supplied pretrained weights are never reset.

    Fusion and predictor also use the recipe's own isolated generators. The
    common ambient seed makes any other constructor RNG usage explicit.
    model_config exists only for the CPU constructor proof; CLI is native-only.
    """
    torch.manual_seed(INITIALIZATION_SEED)
    base = OLMoTiledRTForCausalLM(model_config or OLMoConfig.native_1b(), device="meta", dtype=torch.float32,
        attention_backend="sdpa", attention_precision="mixed", ordinary_activation_checkpointing=True,
        cast_weights_once=True, tile_backend="triton", backward_tile_backend="triton",
        backward_memory="recompute", reuse_rope=True, kv_only_writes=True,
        ordinary_pointwise_backend="eager", ordinary_rope_backend="native")
    base.load_state_dict(state, strict=True, assign=True)
    model = build_campaign_model(base, recipe)
    apply_smoke_objective(model)
    model.to(device).train()
    if model.backbone.readout_weight is not model.backbone.token_embeddings.weight:
        raise ValueError("Original embedding/readout tying was lost")
    return model, build_campaign_adamw(model, recipe, fused=torch.device(device).type == "cuda")


def expected_counts(plan, *, nextlat):
    counts = plan.counts
    if len(plan.rows) != REAL_ROWS or counts.valid_tokens != VALID_TOKENS or counts.packed_rows != REAL_ROWS:
        raise ValueError("Native smoke requires exactly 48 complete T1024 rows, without dummy slots")
    return {"ce": counts.ce_targets, "latent": counts.latent_pairs if nextlat else 0,
            "kl": counts.kl_triples if nextlat else 0}


def validate_inputs(plan, rank_evidence, *, nextlat):
    """Cross-check immutable row membership and independent token-mask counts."""
    expected = expected_counts(plan, nextlat=nextlat)
    if len(rank_evidence) != WORLD_SIZE or any(e["slots"] != SLOTS for e in rank_evidence):
        raise ValueError("Require two ranks with two physical slots each")
    canonical = [row.key for row in plan.rows]
    rows = [row for evidence in rank_evidence for row in evidence["rows"]]
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(canonical):
        raise ValueError("Distributed rows are missing, duplicated, or foreign")
    observed = sum_objective_counts([e["literal_counts"] for e in rank_evidence])
    model_counts = sum_objective_counts([e["model_counts"] for e in rank_evidence])
    if observed != expected or model_counts != expected:
        raise ValueError("Literal masks, model denominators, and packed plan differ")
    if sum(e["input_tokens"] for e in rank_evidence) != VALID_TOKENS:
        raise ValueError("Valid token count differs from the packed plan")
    mapping = {row["key"]: row for row in rows}
    ordered = [mapping[key] for key in canonical]
    return {"counts": expected, "input_tokens": VALID_TOKENS, "documents": REAL_ROWS,
            "microbatches": WORLD_SIZE * SLOTS, "rows": ordered,
            "input_sha256": digest(ordered), "cursor": asdict(plan.start_cursor)}


def validate_update(metrics, evidence, counters, *, update, previous_counters):
    for name in ("counts", "input_tokens", "documents", "microbatches"):
        if metrics[name] != evidence[name]:
            raise ValueError("Executed update accounting differs: " + name)
    if metrics["world_size"] != WORLD_SIZE or metrics["local_microbatches"] != SLOTS:
        raise ValueError("Actual graph accumulation differs")
    values = [metrics["objective"], metrics["gradient_norm_before_clip"],
              *metrics["loss_sums"].values(), *metrics["lr_used"], *metrics["lr_next"]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("Nonfinite update metrics")
    expected = dict(previous_counters)
    expected["optimizer_updates"] += 1
    for name in ("microbatches", "documents", "input_tokens"):
        expected[name] += evidence[name]
    for name, term in (("ce_positions", "ce"), ("latent_pairs", "latent"), ("kl_triples", "kl")):
        expected[name] += evidence["counts"][term]
    if update != expected["optimizer_updates"] or asdict(counters) != expected or metrics["counters"] != expected:
        raise ValueError("Optimizer/token/target counters differ")
    return expected


def tensor_record(value, *, require_fp32=True):
    if not isinstance(value, torch.Tensor) or (require_fp32 and value.dtype != torch.float32):
        raise ValueError("Parameters, gradients and Adam must remain FP32 tensors")
    value = value.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite tensor in state or gradients")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_digest(value)}


def state_audit(model, optimizer, *, update):
    """Stream exact CPU digests; never keep another native model/state copy."""
    ownership = optimizer_ownership(model, optimizer)
    parameters = {n: tensor_record(p) for n, p in model.named_parameters()}
    active = {p: n for n, p in model.named_parameters() if p.requires_grad}
    if update == 0:
        if optimizer.state:
            raise ValueError("Original start must have fresh empty Adam")
    elif set(optimizer.state) != set(active):
        raise ValueError("Adam state must belong to every and only active parameter")
    adam = {}
    for parameter, state in optimizer.state.items():
        if parameter not in active or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unexpected Adam ownership or fields")
        if state["step"].numel() != 1 or float(state["step"]) != update:
            raise ValueError("Adam counter differs from actual updates")
        if any(tuple(state[name].shape) != tuple(parameter.shape) for name in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Adam moment shape differs")
        adam[active[parameter]] = {name: tensor_record(value) for name, value in state.items()}
    return {"parameters": parameters, "adam": adam, "ownership": ownership, "optimizer_updates": update}


def gradient_audit(model):
    record = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise ValueError("Missing active parameter gradient: " + name)
            record[name] = tensor_record(parameter.grad)
        elif parameter.grad is not None:
            raise ValueError("Frozen module acquired a gradient: " + name)
    return record


def changed_components(model, before, after):
    changed = {"backbone": 0, "fusion": 0, "predictor": 0}
    present = set()
    for name, parameter in model.named_parameters():
        component = "predictor" if name.startswith("predictor.") else (
            "fusion" if name.startswith("backbone.fusion.") else "backbone")
        differs = before["parameters"][name] != after["parameters"][name]
        if not parameter.requires_grad and differs:
            raise ValueError("Frozen parameter changed: " + name)
        if parameter.requires_grad:
            present.add(component)
            changed[component] += int(differs)
    if any(changed[name] == 0 for name in present):
        raise ValueError("An active component made no parameter change")
    return changed


def artifact_stats(root):
    paths = [root / MANIFEST_FILENAME] + [root / "native" / name for name in FILE_SPECS]
    return {str(p): [p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def run(args, coordinator, device, report, persist, tracker):
    c = coordinator
    recipe = smoke_recipe(args.arm)
    original_stats = c.call("original artifact stats", lambda: artifact_stats(args.artifacts))
    manifest = c.call("authenticate original checkpoint", lambda: validate_prepared_manifest(args.artifacts))
    report["original_checkpoint"] = manifest["checkpoint"]
    report["original_revision"] = manifest["revision"]
    report["original_manifest_sha256"] = sha256_file(args.artifacts / MANIFEST_FILENAME)
    state = c.call("load original checkpoint", lambda: load_native_state_dict(args.artifacts))
    report["initialization"] = {"common_seed": INITIALIZATION_SEED,
        "fusion_seed": recipe.fusion_seed, "predictor_seed": recipe.predictor_seed,
        "feedback_jitter_seed": recipe.jitter_seed, "backbone": "strict loaded original weights; never reset"}
    c.same("common initialization contract", report["initialization"])
    model, optimizer = c.call("construct original plus fresh modules", lambda: construct_model(state, recipe, device))
    del state
    gc.collect()
    data = c.call("authenticate ordered corpus", lambda: OrderedCampaignData(args.corpus, args.index))
    runner = adapter = None
    try:
        if data.manifest_sha256 != args.index_sha256 or data.length != LENGTH or data.split != "train":
            raise ValueError("Ordered data authority/split/length differs")
        plans, cursor = [], data.cursor()
        for _ in range(UPDATES):
            plan = data.peek_update(cursor, VALID_TOKENS)
            if plan is None:
                raise ValueError("The corpus cannot supply both full smoke updates")
            expected_counts(plan, nextlat=recipe.nextlat)
            plans.append(plan)
            cursor = plan.next_cursor
        schedule = CampaignTokenSchedule(optimizer, [VALID_TOKENS] * UPDATES,
            warmup_tokens=recipe.warmup_tokens, start_fraction=recipe.warmup_start_fraction)
        counters = TrainingCounters()
        recipe_record = recipe.to_dict()
        recipe_record["auxiliary"]["kl"] = 0.1
        report.update(recipe=recipe_record, actual_model_config=model.config.to_dict(),
            objective_weights=model.objective_weights(), mode=asdict(recipe.mode()),
            parameters={"resident": sum(p.numel() for p in model.parameters()),
                        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
            schedule=schedule.checkpoint_contract(), data={"corpus": str(args.corpus), "index": str(args.index),
                "index_sha256": data.manifest_sha256, "start_cursor": asdict(data.cursor()),
                "policy": recipe.document_policy, "extra_eos_or_context_reset": False},
            updates=[], boundaries={})
        def boundary(update):
            value = c.call("finite FP32 model and owned Adam audit", lambda: state_audit(model, optimizer, update=update))
            c.same("exact model/Adam replicas", value)
            return value
        report["boundaries"]["original"] = previous = boundary(0)
        report["optimizer_initial_state_entries"] = len(optimizer.state)
        c.call("tracking native configuration", lambda: tracker.summary({"native/parameters": report["parameters"],
            "native/recipe": recipe_record, "native/objective_weights": model.objective_weights()}), rank_zero=True)
        persist("original_loaded")
        def materialize(index):
            plan = plans[index]
            packed = data.rank_batches(plan, rank=c.rank, world_size=c.world_size, physical_batch_size=PHYSICAL_BATCH)
            noises = tuple(feedback_noise_for_rows(recipe, keys, logical_update=index,
                sequence_length=LENGTH, width=model.config.model_dim, physical_batch_size=PHYSICAL_BATCH)
                for keys in packed.keys)
            rows = []
            for batch, keys in zip(packed.batches, packed.keys):
                if tuple(batch.input_ids.shape) != (PHYSICAL_BATCH, LENGTH) or len(keys) != PHYSICAL_BATCH:
                    raise ValueError("Unexpected packed physical shape or padding")
                for row, key in enumerate(keys):
                    rows.append({"key": key, "input_sha256": tensor_digest(batch.input_ids[row]),
                        "masks_sha256": digest({name: tensor_digest(getattr(batch, name)[row]) for name in
                            ("valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask")})})
            local = {"rows": rows, "slots": len(packed.batches),
                "literal_counts": literal_counts(packed.batches, nextlat=recipe.nextlat),
                "model_counts": sum_objective_counts([model.counts(batch) for batch in packed.batches]),
                "input_tokens": sum(int(batch.valid_mask.sum()) for batch in packed.batches)}
            return packed.batches, noises, local
        batches, noises, local = c.call("capture inputs", lambda: materialize(0))
        evidence = validate_inputs(plans[0], c.gather(local), nextlat=recipe.nextlat)
        adapter = CampaignObjective(model, batches[0], mode=recipe.mode(), global_counts=evidence["counts"],
            world_size=c.world_size, feedback_noise=noises[0],
            config=LMTrainingConfig(precision="bf16_mixed", max_grad_norm=recipe.max_grad_norm))
        runner = CampaignDDPGraphTraining(adapter)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        def phase(name, action):
            c.call("graph observer", lambda: persist("graph/" + name + "/" + action), rank_zero=True)
        runner.prepare(warmup=11, phase_observer=phase)
        runner.capture(warmup=11, release_transient_cache=True, phase_observer=phase)
        report["preparation_seconds_by_rank"] = c.gather(time.monotonic() - started)
        report["boundaries"]["prepared"] = prepared = boundary(0)
        if prepared != previous or not c.same("zero prepared gradients", gradients_are_zero(model)):
            raise ValueError("Graph preparation changed original state")
        pointers, graph_ids = pointer_snapshot(runner), (id(runner.local_graph), id(runner.sync_graph))
        report["memory_after_capture_by_rank"] = c.gather(memory())
        persist("prepared")
        del batches, noises, local
        previous_keys = set()
        previous_token_digest = None
        for index, plan in enumerate(plans):
            batches, noises, local = c.call("changed update inputs", lambda: materialize(index))
            evidence = validate_inputs(plan, c.gather(local), nextlat=recipe.nextlat)
            keys = {row["key"] for row in evidence["rows"]}
            if keys & previous_keys:
                raise ValueError("Changed-input updates reused a logical row")
            previous_keys |= keys
            token_digest = digest([row["input_sha256"] for row in evidence["rows"]])
            if token_digest == previous_token_digest:
                raise ValueError("The second update did not change input tokens")
            previous_token_digest = token_digest
            previous_counters = asdict(counters)
            started = time.monotonic()
            result = runner.backward(batches, feedback_noises=noises, replay=True)
            gradients = c.call("finite active raw gradient audit", lambda: gradient_audit(model))
            c.same("exact raw gradient replicas", gradients)
            metrics = runner.step(result, optimizer, scheduler=schedule, counters=counters)
            validate_update(metrics, evidence, counters, update=index + 1, previous_counters=previous_counters)
            after = boundary(index + 1)
            changed = c.call("active component changes", lambda: changed_components(model, previous, after))
            c.same("component changes", changed)
            same_pointers = pointers == pointer_snapshot(runner)
            same_graphs = graph_ids == (id(runner.local_graph), id(runner.sync_graph))
            if not all(c.gather(same_pointers and same_graphs and gradients_are_zero(model))):
                raise ValueError("Captured storage/graphs or completed gradient boundary changed")
            data.commit(plan.start_cursor, plan)
            if schedule.completed_tokens != counters.input_tokens or asdict(data.cursor()) != asdict(plan.next_cursor):
                raise ValueError("Scheduler/data clocks differ after actual update")
            report["boundaries"]["update" + str(index + 1)] = after
            row = {"update": index + 1, "metrics": metrics, "input": evidence,
                "raw_gradient_sha256": digest(gradients), "raw_gradient_tensors": len(gradients),
                "exact_gradient_replicas": True, "exact_state_replicas": True, "finite_fp32_state": True,
                "active_components_changed": changed, "same_graph_objects": same_graphs,
                "stable_storage": same_pointers, "memory_by_rank": c.gather(memory()),
                "wall_seconds_including_cpu_audits": time.monotonic() - started}
            report["updates"].append(row)
            report["final_counters"] = asdict(counters)
            def log():
                tracker.log({"update": index + 1, "train/objective": metrics["objective"],
                    "train/raw_gradient_norm": metrics["gradient_norm_before_clip"],
                    "train/lr": metrics["lr_used"][0],
                    **{"train/mean_" + term: metrics["loss_sums"][term] / max(1, metrics["counts"][term])
                       for term in ("ce", "latent", "kl")}}, step=index + 1)
                print({"arm": args.arm, "update": index + 1, "objective": metrics["objective"],
                       "raw_gradient_norm": metrics["gradient_norm_before_clip"], "checks": "passed"}, flush=True)
            c.call("tracking update", log, rank_zero=True)
            persist("update/" + str(index + 1))
            previous = after
            del batches, noises, local, gradients
        report["runner_by_rank"] = c.gather(runner.metadata)
        if any(r["local_replay_calls"] != UPDATES or r["sync_replay_calls"] != UPDATES
               for r in report["runner_by_rank"]):
            raise ValueError("Actual local/synchronized graph replay counts differ")
        data.validate_integrity()
        if artifact_stats(args.artifacts) != original_stats:
            raise ValueError("Original artifact files changed during smoke")
        report["original_files_unchanged"] = True
    finally:
        data.close()
    # On success, destroy graph captures and DDP reducer before the NCCL group.
    # A failed collective is handled by the external whole-job timeout, never a
    # new cleanup collective from a potentially desynchronized rank.
    if runner is not None:
        release_completed_graph_runner(runner)
    del runner, adapter, optimizer, model
    gc.collect()
    torch.cuda.synchronize(device)


def main(argv=None):
    args = parse_args(argv)
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Use the project Docker GPU launcher")
    if os.environ.get("WORLD_SIZE") != "2":
        raise RuntimeError("Native smoke requires exactly two torchrun ranks")
    if any(os.environ.get(k) != "0" for k in ("NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING")):
        raise RuntimeError("Captured NCCL requires both async-error flags0 and an external whole-job timeout")
    device, runtime, determinism = configure_cuda_runtime(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", timeout=timedelta(seconds=900), device_id=device)
    c = Coordinator()
    report = {"schema": SCHEMA, "status": "running", "arm": args.arm, "runtime": runtime,
        "determinism": determinism, "sources": source_hashes(), "execution": EXECUTION,
        "world_size": WORLD_SIZE, "sequence_length": LENGTH, "physical_batch_per_rank": PHYSICAL_BATCH,
        "slots_per_rank": SLOTS, "real_rows_per_update": REAL_ROWS, "valid_tokens_per_update": VALID_TOKENS,
        "actual_updates": UPDATES, "checkpoint_policy": "No disposable full checkpoint; evidence only",
        "scope": "Functionality only: original pretrained backbone + fresh modules/Adam, packed changed inputs, exact counts and replicas. Not performance, quality, BF16 equivalence or native restart."}
    tracker, error = None, None
    output_owned = False
    started = time.monotonic()
    def persist(stage=None):
        if stage is not None:
            report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - started
        if c.rank == 0:
            report["wandb"] = None if tracker is None else tracker.record
            write_json(args.output_dir / "report.json", report)
    try:
        c.same("source pins", report["sources"])
        def setup():
            nonlocal tracker, output_owned
            args.output_dir.mkdir(parents=True, exist_ok=False)
            output_owned = True
            for name, pin in report["sources"].items():
                target = args.output_dir / "source-snapshot" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
                if sha256_file(target) != pin:
                    raise ValueError("Source changed during snapshot")
            tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                name=args.output_dir.name, group="olmo-topology-native-smoke", preserve_state=preserve_local_rng)
            tracker.start({k: report[k] for k in ("arm", "execution", "scope", "world_size", "real_rows_per_update")})
            print({"wandb": tracker.record["run_url"]}, flush=True)
        c.call("owned output and online tracking", setup, rank_zero=True)
        persist("setup")
        with disable_autocast_weight_cache(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            run(args, c, device, report, persist, tracker)
        if source_hashes() != report["sources"]:
            raise ValueError("Runtime sources changed")
        dist.destroy_process_group()
        report.update(status="completed", teardown="completed")
    except BaseException as exc:
        error = exc
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc),
                                              "traceback": traceback.format_exc()})
        raise
    finally:
        if c.rank == 0 and output_owned:
            persist()
            try:
                if tracker is not None:
                    tracker.finish(succeeded=error is None)
            except BaseException as exc:
                report.update(status="failed", tracking_error={"type": type(exc).__name__, "message": str(exc)})
                raise
            finally:
                persist()


if __name__ == "__main__":
    main()
