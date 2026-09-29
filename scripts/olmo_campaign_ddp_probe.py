#!/usr/bin/env python3
"""Bounded real two-GPU campaign DDP/graph acceptance, never quality training.

Run with torchrun --nproc_per_node=2 in the project GPU container. The external
launcher must have a timeout and both NCCL async-error environment flags set to
0. Each process first runs a local global-batch reference on its own GPU, then
reuses the same parameter storage for the distributed run. The default is the
independent canonical objective. Prepared-reference mode isolates DDP behavior
and retains canonical-versus-prepared arithmetic as a separate qualification.
No full diagnostic model/optimizer checkpoint is written by this probe.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gc
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
from cdrm.pretrained.campaign_recipe import (ARMS, CampaignRecipe, CampaignTokenSchedule,
    build_campaign_model, build_campaign_adamw, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective, CampaignGraphTraining
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import (load_native_state_dict, load_native_tokenizer,
    validate_prepared_manifest)
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_graph_probe import (compare_metrics, compare_optimizer,
    compare_tensor_mappings, gradients_are_zero, optimizer_snapshot, parameter_snapshot,
    pointer_snapshot, restore_parameters_in_place, rng_snapshot, rng_unchanged)
from scripts.olmo_campaign_probe import gradient_record, memory
from scripts.olmo_lm_common import tensor_digest
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

TERMS = ("ce", "latent", "kl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("tiny", "pretrained"), required=True)
    parser.add_argument("--case", choices=("eager", "graph"), required=True)
    parser.add_argument("--document-policy", choices=("isolated-v1", "continuous-stream-v1"),
                        default="isolated-v1")
    parser.add_argument("--reference", choices=("canonical", "prepared"), default="canonical",
                        help="Local reference arithmetic; prepared separately retains canonical-vs-dense qualification")
    parser.add_argument("--arms", help="Comma-separated campaign arms; tiny defaults to all eight, pretrained B,NFR")
    parser.add_argument("--length", type=int)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmup", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    args.length = args.length or (8 if args.scale == "tiny" else 16)
    if args.length not in ((8, 16) if args.scale == "tiny" else (16, 32)):
        parser.error("Bounded probe requires tiny T8/T16 or pretrained T16/T32")
    if args.warmup < 11:
        parser.error("DDP capture preparation requires at least 11 warmup backward calls")
    args.arms = tuple(args.arms.split(",")) if args.arms else (ARMS if args.scale == "tiny" else ("B", "NFR"))
    if not args.arms or len(set(args.arms)) != len(args.arms) or any(a not in ARMS for a in args.arms):
        parser.error("Arms must be distinct campaign names")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT.resolve()):
        parser.error("Evidence must remain under the persistent project checkout")
    return args


def fixture_for_update(recipe, width, rank, update, *, length, token_ids, eos_id, batch_size=2):
    """M1/M2/M3 global updates with unequal, empty-rank and final-empty slots.

    The canonical reference consumes exactly the same physical rows and noise
    as DDP. It does not concatenate rows into a different GEMM shape. Dummy rows
    are suffix rows, as required by feedback_noise_for_rows' padding contract.
    """
    if rank not in (0, 1) or update not in (0, 1, 2) or length < 8 or batch_size not in (1, 2):
        raise ValueError("Fixture requires two ranks, update 0..2, T>=8 and B1/B2")
    if not token_ids or any(type(t) is not int or t < 0 for t in token_ids):
        raise ValueError("Fixture source must contain nonnegative integer tokens")
    sizes = (
        (((length, 5),), ((6, 2),)),
        (((length-1, 4), (5, 0)), ((0, 0), (0, 0))),
        (((length-2, 6), (4, 0), (0, 0)), ((7, 3), (3, 0), (0, 0))),
    )[update][rank]
    batches, noises = [], []
    for microbatch, lengths in enumerate(sizes):
        ids = torch.full((batch_size, length), eos_id, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        docs = torch.full_like(ids, -1)
        keys = []
        for row, size in enumerate(lengths[:batch_size]):
            if not size:
                continue
            offset = update*17 + rank*11 + microbatch*5 + row*3
            content = [token_ids[(offset+i) % len(token_ids)] for i in range(size-1)] + [eos_id]
            ids[row, :size] = torch.tensor(content)
            valid[row, :size] = True
            docs[row, :size] = update*100 + rank*20 + microbatch*2 + row
            if getattr(recipe, "document_policy", "isolated-v1") == "continuous-stream-v1" and size >= 3:
                # True metadata boundaries move between updates. EOS tokens do
                # not themselves define boundaries in the model/loss policy.
                boundary = 1 + (update + rank + microbatch + row) % (size - 1)
                ids[row, boundary-1] = eos_id
                docs[row, :size] *= 2
                docs[row, boundary:size] += 1
            keys.append(f"ddp-u{update}-r{rank}-m{microbatch}-row{row}")
        ce, latent, kl = (valid.clone() for _ in TERMS)
        if update:
            ce[:, 4::3] = False
            latent[:, 2::3] = False
            kl[:, 3::4] = False
        batches.append(NextLatBatch(ids, valid, docs, ce, latent, kl))
        noises.append(feedback_noise_for_rows(recipe, keys, logical_update=update,
            sequence_length=length, width=width, physical_batch_size=batch_size))
    return tuple(batches), tuple(noises)


def construct(args, arm, device):
    """Deterministic model/recipe/fixture source, shared with fresh-restart probe."""
    torch.manual_seed(20260929)
    tiny = args.scale == "tiny"
    recipe = CampaignRecipe(arm, sequence_length=args.length if tiny else 1024,
                            rt_layers=(0, 1) if tiny else (0, 15),
                            document_policy=getattr(args, "document_policy", "isolated-v1"))
    if tiny:
        base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
            attention_precision="fp32", tile_backend="eager", backward_tile_backend="eager",
            backward_memory="recompute", reuse_rope=True, kv_only_writes=True)
        checkpoint, ids, eos = {"fixture": "seeded tiny model; no pretrained claims"}, list(range(2, 59)), 60
    else:
        manifest = validate_prepared_manifest(args.artifacts)
        checkpoint = manifest["checkpoint"]
        state = load_native_state_dict(args.artifacts)
        base = OLMoTiledRTForCausalLM(OLMoConfig.native_1b(), device="meta", dtype=torch.float32,
            attention_backend="sdpa", attention_precision="mixed", ordinary_activation_checkpointing=True,
            cast_weights_once=True, tile_backend="triton", backward_tile_backend="triton",
            backward_memory="recompute", reuse_rope=True, kv_only_writes=True)
        base.load_state_dict(state, strict=True, assign=True)
        del state
        tokenizer = load_native_tokenizer(args.artifacts)
        ids = tokenizer.encode("The model keeps a record of earlier tokens and predicts the next token. "*20)
        eos = tokenizer.eos_token_id
    model = build_campaign_model(base, recipe).to(device).train()
    return model, recipe, checkpoint, ids, eos


def source_hashes():
    helpers = ("olmo_campaign_ddp_probe.py", "olmo_campaign_graph_probe.py", "olmo_campaign_probe.py",
               "olmo_two_gpu_validate.py", "olmo_lm_common.py", "experiment_tracking.py", "olmo_validation.py")
    files = set((ROOT / "cdrm/pretrained").rglob("*.py")) | {ROOT/"scripts"/name for name in helpers}
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(files)}


def global_fixture_metadata(model, fixtures):
    batches = tuple(batch for rank_batches, _ in fixtures for batch in rank_batches)
    return {"counts": sum_objective_counts([model.counts(batch) for batch in batches]),
            "microbatches": len(batches), "documents": sum(int(b.valid_mask.any(-1).sum()) for b in batches),
            "input_tokens": sum(int(b.valid_mask.sum()) for b in batches)}


def advance_counters(counters, metrics):
    counters.optimizer_updates += 1
    for key in ("microbatches", "documents", "input_tokens"):
        setattr(counters, key, getattr(counters, key) + metrics[key])
    for attr, term in (("ce_positions", "ce"), ("latent_pairs", "latent"), ("kl_triples", "kl")):
        setattr(counters, attr, getattr(counters, attr) + metrics["counts"][term])


def canonical_backward(model, recipe, fixtures, *, precision):
    """Ordinary sparse model objective, independent of CampaignObjective/layout."""
    model.zero_grad(set_to_none=True)
    metrics = global_fixture_metadata(model, fixtures)
    totals, objective_total = dict.fromkeys(TERMS, 0.0), 0.0
    device = next(model.parameters()).device
    for batches, noises in fixtures:
        for batch, noise in zip(batches, noises):
            noise = None if noise is None else tuple(v.to(device) for v in noise)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16_mixed", cache_enabled=False):
                result = model.loss_sums(batch.to(device), backbone_kwargs={"mode": recipe.mode(),
                    "feedback_noise": noise, "right_padded_causal": True})
                # Assemble literal campaign per-pass coefficients independently.
                count = len(result.pass_losses)
                coefficients = (1.0,) if count == 1 else (.5,) + (.5/(count-1),)*(count-1)
                sums = {t: sum(p.sums[t] * (coefficients[i] if t == "ce" else 1/count)
                              for i, p in enumerate(result.pass_losses)) for t in TERMS}
                objective = sum(sums[t] * result.weights[t] / metrics["counts"][t]
                                for t in TERMS if result.weights[t])
            objective.backward()
            objective_total += float(objective.detach())
            for t in TERMS:
                totals[t] += float(sums[t].detach())
    return {**metrics, "loss_sums": totals, "objective": objective_total}


def prepared_backward(model, recipe, fixtures, *, precision):
    """Single-device dense objective, no DDP/reducer and no captured graph.

    Every physical microbatch retains exactly its original B/T and keyed noise.
    The local runner uses world_size=1 and the full global-update denominators;
    its initialization backward is cleared before collecting the reference.
    """
    model.zero_grad(set_to_none=True)
    batches = tuple(batch for rank_batches, _ in fixtures for batch in rank_batches)
    noises = tuple(noise for _, rank_noises in fixtures for noise in rank_noises)
    metadata = global_fixture_metadata(model, fixtures)
    adapter = CampaignObjective(model, batches[0], mode=recipe.mode(),
        global_counts=metadata["counts"], world_size=1, feedback_noise=noises[0],
        config=LMTrainingConfig(precision=precision, max_grad_norm=recipe.max_grad_norm))
    return CampaignGraphTraining(adapter).backward(batches, feedback_noises=noises, replay=False)


def state_digests(model, optimizer, scheduler, counters):
    """Exact inter-rank agreement; bounded one-tensor-at-a-time host copies."""
    result = {"parameters": {n: tensor_digest(p) for n, p in model.named_parameters()},
              "optimizer": {}, "scheduler": scheduler.state_dict(), "counters": asdict(counters)}
    for name, parameter in model.named_parameters():
        result["optimizer"][name] = {key: tensor_digest(value) if isinstance(value, torch.Tensor) else value
            for key, value in optimizer.state.get(parameter, {}).items()}
    return result


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def run_arm(args, arm, rank, publish):
    from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
    device = torch.device("cuda", rank)
    model, recipe, checkpoint, ids, eos = construct(args, arm, device)
    precision = "fp32" if args.scale == "tiny" else "bf16_mixed"
    fixtures = [[fixture_for_update(recipe, model.config.model_dim, r, u, length=args.length,
                  token_ids=ids, eos_id=eos, batch_size=args.batch_size) for r in range(2)] for u in range(3)]
    metadata = [global_fixture_metadata(model, f) for f in fixtures]
    tokens = [m["input_tokens"] for m in metadata]
    initial = parameter_snapshot(model)
    context = nullcontext() if args.scale == "tiny" else sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    with context:
        if args.reference == "prepared":
            before = rng_snapshot()
            canonical = canonical_backward(model, recipe, fixtures[0], precision=precision)
            canonical_gradients, canonical_snapshot = gradient_record(model, save_cpu=True)
            prepared = prepared_backward(model, recipe, fixtures[0], precision=precision)
            qualification_gradients, _ = gradient_record(model, canonical_snapshot)
            qualification_gradients["comparison"]["scope"] = (
                "Fixed initial parameters/global physical rows/noise: canonical selected-position losses "
                "versus prepared dense-mask losses, both single-device without DDP")
            metric_check = compare_metrics(prepared, canonical)
            healthy = canonical_gradients["finite"] and qualification_gradients["finite"] and rng_unchanged(before)
            publish(arm, "canonical-versus-prepared-qualification", {
                "canonical_metrics": canonical, "prepared_metrics": prepared,
                "canonical_gradients": canonical_gradients, "gradient_comparison": qualification_gradients,
                "metric_comparison": metric_check, "rng_unchanged": rng_unchanged(before),
                "finite_execution": healthy,
                "qualification": "Separate unchanged-budget arithmetic compatibility check; a failure remains unresolved even if prepared-reference distributed execution passes",
                "passed": healthy and metric_check["passed"]
                    and qualification_gradients["comparison"]["all_parameters_close"]}, gating=False)
            if not healthy:
                raise AssertionError("Canonical/prepared localization produced nonfinite gradients or changed RNG")
            del canonical_snapshot
            model.zero_grad(set_to_none=True)
            restore_parameters_in_place(model, initial)
            gc.collect()
        reference_backward = canonical_backward if args.reference == "canonical" else prepared_backward
        optimizer = build_campaign_adamw(model, recipe, fused=True)
        schedule = CampaignTokenSchedule(optimizer, tokens, warmup_tokens=recipe.warmup_tokens,
                                         start_fraction=recipe.warmup_start_fraction)
        counters = TrainingCounters()
        references = []
        for update, fixture in enumerate(fixtures):
            before = rng_snapshot()
            metrics = reference_backward(model, recipe, fixture, precision=precision)
            gradients, snapshot = gradient_record(model, save_cpu=True)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            advance_counters(counters, metrics)
            references.append((metrics, snapshot))
            publish(arm, f"{args.reference}-update-{update+1}", {"metrics": metrics, "gradients": gradients,
                "preclip_norm": norm, "rng_unchanged": rng_unchanged(before), "memory": memory(),
                "passed": gradients["finite"] and rng_unchanged(before)})
        expected_parameters = parameter_snapshot(model)
        expected_optimizer = optimizer_snapshot(model, optimizer)
        expected_scheduler, expected_counters = schedule.state_dict(), asdict(counters)
        del optimizer, schedule, counters
        restore_parameters_in_place(model, initial)
        model.zero_grad(set_to_none=True)
        gc.collect()
        first_batches, first_noises = fixtures[0][rank]
        adapter = CampaignObjective(model, first_batches[0], mode=recipe.mode(),
            global_counts=metadata[0]["counts"], world_size=2, feedback_noise=first_noises[0],
            config=LMTrainingConfig(precision=precision, max_grad_norm=recipe.max_grad_norm))
        runner = CampaignDDPGraphTraining(adapter)
        before = rng_snapshot()
        if args.case == "graph":
            runner.capture(warmup=args.warmup)
        else:
            runner.prepare(warmup=args.warmup)
        zeros = gradients_are_zero(model)
        pointers = pointer_snapshot(runner)
        untouched = compare_tensor_mappings(dict(model.named_parameters()), initial, atol=0, rtol=0)["passed"]
        publish(arm, "prepare", {"warmup_backward_calls": runner.warmup_backward_calls,
            "capture_backward_calls": runner.capture_backward_calls,
            "rng_unchanged": rng_unchanged(before), "gradients_zero": zeros, "memory": memory(),
            "checkpoint": checkpoint, "recipe": recipe.to_dict(), "fixture": metadata,
            "parameters_unchanged": untouched, "optimizer_updates": 0,
            "passed": zeros and untouched and rng_unchanged(before) and runner.warmup_backward_calls >= args.warmup})
        optimizer = build_campaign_adamw(model, recipe, fused=True)
        schedule = CampaignTokenSchedule(optimizer, tokens, warmup_tokens=recipe.warmup_tokens,
                                         start_fraction=recipe.warmup_start_fraction)
        counters = TrainingCounters()
        for update, fixture in enumerate(fixtures):
            batches, noises = fixture[rank]
            before = rng_snapshot()
            metrics = runner.backward(batches, feedback_noises=noises, replay=args.case == "graph")
            gradients, _ = gradient_record(model, references[update][1])
            metric_check = compare_metrics(metrics, references[update][0])
            publish(arm, f"distributed-raw-update-{update+1}", {"metrics": metrics,
                "gradient_comparison": gradients, "metric_comparison": metric_check,
                "rng_unchanged": rng_unchanged(before), "pointers_stable": pointers == pointer_snapshot(runner),
                "passed": gradients["finite"] and gradients["comparison"]["all_parameters_close"]
                    and metric_check["passed"] and rng_unchanged(before) and pointers == pointer_snapshot(runner)})
            before = rng_snapshot()
            row = runner.step(metrics, optimizer, scheduler=schedule, counters=counters)
            torch.cuda.synchronize()
            row.update(rng_unchanged=rng_unchanged(before), gradients_zero=gradients_are_zero(model),
                       pointers_stable=pointers == pointer_snapshot(runner), memory=memory())
            row["passed"] = row["rng_unchanged"] and row["gradients_zero"] and row["pointers_stable"]
            publish(arm, f"distributed-adam-update-{update+1}", row)
            # Free each billion-scale reference gradient as soon as consumed.
            references[update] = None
        parameter_check = compare_tensor_mappings(dict(model.named_parameters()), expected_parameters,
                                                   atol=3e-6, rtol=3e-5, initial=initial)
        optimizer_check = compare_optimizer(model, optimizer, expected_optimizer)
        schedule_equal = schedule.state_dict() == expected_scheduler
        counters_equal = asdict(counters) == expected_counters
        digests = gather(state_digests(model, optimizer, schedule, counters))
        ranks_equal = digests[0] == digests[1]
        publish(arm, "three-update-parity", {"parameters": parameter_check, "optimizer": optimizer_check,
            "scheduler_equal": schedule_equal, "counters_equal": counters_equal, "ranks_exact": ranks_equal,
            "counters": asdict(counters), "graph": {"warmup_backward_calls": runner.warmup_backward_calls,
                "capture_backward_calls": runner.capture_backward_calls, "replay_calls": runner.replay_calls},
            "passed": parameter_check["passed"] and optimizer_check["passed"] and schedule_equal
                      and counters_equal and ranks_equal})
    del runner, adapter, optimizer, model, references, expected_parameters, expected_optimizer, initial
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None):
    args = parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("This acceptance probe requires exactly two torchrun ranks")
    if any(os.environ.get(name) != "0" for name in ("NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING")):
        raise RuntimeError("Set both NCCL async-error flags to 0 and use an external timeout")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=torch.device("cuda", rank))
    tracker = None
    started = time.monotonic()
    report = {"schema": "olmo-campaign-two-gpu-probe-v1", "status": "running", "operational_status": "running",
        "independent_reference_status": "not_completed", "rank": rank,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "scale": args.scale, "case": args.case, "reference": args.reference, "arms": args.arms, "length": args.length,
        "document_policy": args.document_policy,
        "physical_batch_per_rank": args.batch_size, "sources": source_hashes(), "rows": [],
        "scope": "Real NCCL campaign accumulation/graph and three Adam updates versus the explicitly named local reference, using the explicit document policy and moving synthetic document boundaries when packed. Prepared reference retains canonical-versus-dense arithmetic as a separate unchanged-budget qualification. Not throughput, real-data loader recovery or FP32 qualification of pretrained BF16",
        "qualification_failures": [],
        "snapshots": "CPU RAM only; atomic small evidence every stage; bounded launcher required"}
    try:
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            for relative in report["sources"]:
                destination = args.output_dir / "source-snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                group="olmo-campaign-two-gpu-readiness", name=args.output_dir.name, preserve_state=preserve_local_rng)
            tracker.start({k: report[k] for k in ("scale", "case", "reference", "arms", "length", "physical_batch_per_rank", "document_policy", "scope")})
            report["wandb"] = tracker.record
            print({"wandb": tracker.record["run_url"]}, flush=True)
        dist.barrier()

        def publish(arm, stage, row, *, gating=True):
            report["rows"].append({"arm": arm, "stage": stage, "gating": gating, **row})
            report["elapsed_seconds"] = time.monotonic()-started
            write_json(args.output_dir / f"rank-{rank}.json", report)
            rows = gather({"rank": rank, **row})
            passed = all(r.get("passed", False) for r in rows)
            if not passed and not gating:
                report["qualification_failures"].append({"arm": arm, "stage": stage,
                    "status": "unresolved", "gating": False})
            if not gating:
                report["independent_reference_status"] = "failed" if report["qualification_failures"] else "passed"
            elif not passed and args.reference == "canonical":
                report["independent_reference_status"] = "failed"
            if rank == 0:
                report["rows"][-1] = {"arm": arm, "stage": stage, "gating": gating, "passed": passed, "ranks": rows}
                write_json(args.output_dir / "report.json", report)
                tracker.log(scalar_metrics({"ranks": {str(r["rank"]): r for r in rows}}, f"diagnostic/{arm}/{stage}"),
                            step=len(report["rows"]))
                print({"arm": arm, "stage": stage, "passed": passed, "gating": gating,
                       "elapsed_seconds": time.monotonic()-started}, flush=True)
            if not passed and gating:
                raise AssertionError(f"Campaign two-GPU probe failed: {arm}/{stage}")

        for arm in args.arms:
            run_arm(args, arm, rank, publish)
        if report["sources"] != source_hashes():
            raise AssertionError("Runtime sources changed during probe")
        report["status"] = report["operational_status"] = "passed"
        if args.reference == "canonical":
            report["independent_reference_status"] = "passed"
    except Exception as error:
        report.update(status="failed", operational_status="failed", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        if args.output_dir.exists():
            write_json(args.output_dir / f"rank-{rank}.json", report)
        if tracker is not None:
            try:
                tracker.finish(succeeded=report["status"] == "passed")
            finally:
                report["wandb"] = tracker.record
                write_json(args.output_dir / "report.json", report)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
