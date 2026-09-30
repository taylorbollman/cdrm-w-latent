#!/usr/bin/env python3
"""Tiny same-precision acceptance for changing campaign rank allocation.

The independent oracle consumes the same ordered logical rows on one device,
without DDP, capture, prepared loss layouts, or padded distributed slots. This
is a bounded correctness probe, not checkpoint migration or a BF16/FP32 study.
Launch CUDA only through the project container and torchrun (one or two ranks).
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import timedelta
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.campaign_data import DataCursor, DocumentWindow, LogicalUpdate, partition_update
from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
from cdrm.pretrained.campaign_recipe import (ARMS, CampaignRecipe, CampaignTokenSchedule,
    build_campaign_adamw, build_campaign_model, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_campaign_graph_probe import (compare_tensor_mappings, gradients_are_zero,
    optimizer_snapshot, parameter_snapshot, pointer_snapshot)
from scripts.olmo_validation import require_container_gpu

TERMS = ("ce", "latent", "kl")
ROWS = (5, 7, 3)


def logical_fixture(update):
    """Changing row counts expose unequal rank work and a dummy final sync slot."""
    if type(update) is not int or not 0 <= update < len(ROWS):
        raise ValueError("Fixture update must be 0, 1, or 2")
    rows = tuple(DocumentWindow(f"allocation-u{update}-row{i}", "train", i, 0,
        tuple(2 + (update * 11 + i * 7 + t) % 55 for t in range(5 + (i + update) % 4)))
        for i in range(ROWS[update]))
    return LogicalUpdate(rows, DataCursor("0" * 64, "train", update * 10),
        DataCursor("0" * 64, "train", update * 10 + len(rows)), sum(r.length for r in rows))


def materialize(rows, *, physical_batch_size=2, length=8):
    """Include true packed-document boundaries and different masks per loss."""
    ids = torch.full((physical_batch_size, length), 60, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    docs = torch.full_like(ids, -1)
    for index, row in enumerate(rows):
        ids[index, :row.length] = torch.tensor(row.tokens)
        valid[index, :row.length] = True
        docs[index, :row.length] = row.document_index * 2
        boundary = 2 + row.document_index % 2
        docs[index, boundary:row.length] += 1
        ids[index, boundary - 1] = 60
    ce, latent, kl = (valid.clone() for _ in TERMS)
    ce[:, 4] = False
    latent[:, 3] = False
    kl[:, 5] = False
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def fixture_for_rank(recipe, update, *, rank, world_size, physical_batch_size=2):
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("Rank must belong to the requested world")
    slots = partition_update(logical_fixture(update), world_size=world_size,
                             physical_batch_size=physical_batch_size)
    row_groups = tuple(slot[rank] for slot in slots)
    batches = tuple(materialize(rows, physical_batch_size=physical_batch_size) for rows in row_groups)
    noises = tuple(feedback_noise_for_rows(recipe, [r.key for r in rows], logical_update=update,
        sequence_length=8, width=32, physical_batch_size=physical_batch_size) for rows in row_groups)
    return batches, noises, tuple(tuple(r.key for r in rows) for rows in row_groups)


def literal_counts(batches, *, nextlat=True):
    """Read eligibility token by token, independently of prepared mask code."""
    counts = dict.fromkeys(TERMS, 0)
    for batch in batches:
        for row in range(batch.input_ids.shape[0]):
            for target in range(1, batch.input_ids.shape[1]):
                if not (batch.valid_mask[row, target - 1] and batch.valid_mask[row, target]):
                    continue
                counts["ce"] += int(batch.ce_mask[row, target])
                pair = batch.document_ids[row, target] == batch.document_ids[row, target - 1]
                counts["latent"] += int(pair and batch.latent_mask[row, target])
                if target >= 2:
                    triple = (pair and batch.valid_mask[row, target - 2]
                        and batch.document_ids[row, target] == batch.document_ids[row, target - 2])
                    counts["kl"] += int(triple and batch.kl_mask[row, target])
    if not nextlat:
        counts.update(latent=0, kl=0)
    return counts


def construct(arm, device):
    torch.manual_seed(20261001)
    recipe = CampaignRecipe(arm, sequence_length=8, rt_layers=(0, 1),
                            warmup_tokens=50, document_policy="continuous-stream-v1")
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", tile_backend="eager", backward_tile_backend="eager",
        backward_memory="recompute", reuse_rope=True, kv_only_writes=True)
    return recipe, build_campaign_model(base, recipe).to(device).train()


def oracle_backward(model, recipe, update):
    """Uncaptured global objective; no DDP normalization or distributed padding."""
    batches, noises, _ = fixture_for_rank(recipe, update, rank=0, world_size=1)
    counts = literal_counts(batches, nextlat=recipe.nextlat)
    sums = dict.fromkeys(TERMS, 0.0)
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    for batch, noise in zip(batches, noises):
        noise = None if noise is None else tuple(v.to(device) for v in noise)
        result = model.loss_sums(batch.to(device), backbone_kwargs={"mode": recipe.mode(),
            "feedback_noise": noise, "right_padded_causal": True})
        n = len(result.pass_losses)
        coefficients = (1.0,) if n == 1 else (0.5,) + (0.5 / (n - 1),) * (n - 1)
        terms = {term: sum(p.sums[term] * (coefficients[i] if term == "ce" else 1 / n)
                 for i, p in enumerate(result.pass_losses)) for term in TERMS}
        sum(terms[t] * result.weights[t] / counts[t] for t in TERMS if result.weights[t]).backward()
        for term in TERMS:
            sums[term] += float(terms[term].detach())
    return {"counts": counts, "loss_sums": sums, "documents": ROWS[update],
            "input_tokens": sum(int(b.valid_mask.sum()) for b in batches)}


def gradients(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def compare_adam(actual, reference):
    def flatten(values):
        return {f"{name}/{key}": value for name, state in values.items()
                for key, value in state.items() if isinstance(value, torch.Tensor)}
    return compare_tensor_mappings(flatten(actual), flatten(reference), atol=5e-7, rtol=1e-4)


def run_acceptance(arm, *, device, case, warmup=11, publish=None):
    """Process group is owned by the caller; return evidence and a tiny state."""
    device = torch.device(device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    recipe, model = construct(arm, device)
    reference = copy.deepcopy(model)
    initial = parameter_snapshot(model)
    batches, noises, _ = fixture_for_rank(recipe, 0, rank=rank, world_size=world_size)
    canonical_batches = fixture_for_rank(recipe, 0, rank=0, world_size=1)[0]
    adapter = CampaignObjective(model, batches[0], mode=recipe.mode(),
        global_counts=literal_counts(canonical_batches, nextlat=recipe.nextlat),
        world_size=world_size, feedback_noise=noises[0],
        config=LMTrainingConfig(precision="fp32", max_grad_norm=recipe.max_grad_norm))
    runner = CampaignDDPGraphTraining(adapter, bucket_cap_mb=0.02)
    if case == "graph":
        runner.capture(warmup=warmup)
    elif case == "eager":
        runner.prepare(warmup=warmup)
    else:
        raise ValueError("Case must be eager or graph")
    pointers = pointer_snapshot(runner)
    preparation_passed = gradients_are_zero(model) and compare_tensor_mappings(
        dict(model.named_parameters()), initial, atol=0, rtol=0)["passed"]
    optim = build_campaign_adamw(model, recipe, fused=device.type == "cuda")
    oracle_optim = build_campaign_adamw(reference, recipe, fused=device.type == "cuda")
    tokens = [logical_fixture(u).counts.presented_tokens for u in range(len(ROWS))]
    schedule = CampaignTokenSchedule(optim, tokens, warmup_tokens=50)
    oracle_schedule = CampaignTokenSchedule(oracle_optim, tokens, warmup_tokens=50)
    counters = TrainingCounters()
    rows = []
    for update in range(len(ROWS)):
        expected = oracle_backward(reference, recipe, update)
        batches, noises, keys = fixture_for_rank(recipe, update, rank=rank, world_size=world_size)
        actual = runner.backward(batches, feedback_noises=noises, replay=case == "graph")
        gradient_check = compare_tensor_mappings(gradients(model), gradients(reference),
                                                 atol=8e-6, rtol=8e-4)
        count_match = all(actual[k] == expected[k] for k in ("counts", "documents", "input_tokens"))
        loss_match = all(math.isclose(actual["loss_sums"][t], expected["loss_sums"][t],
                                     abs_tol=1e-5, rel_tol=3e-6) for t in TERMS)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), recipe.max_grad_norm, foreach=False)
        oracle_optim.step()
        oracle_schedule.step()
        step = runner.step(actual, optim, scheduler=schedule, counters=counters)
        parameter_check = compare_tensor_mappings(dict(model.named_parameters()),
            parameter_snapshot(reference), atol=3e-6, rtol=3e-5, initial=initial)
        adam_check = compare_adam(optimizer_snapshot(model, optim), optimizer_snapshot(reference, oracle_optim))
        scheduler_match = schedule.state_dict() == oracle_schedule.state_dict()
        pointer_match = pointers == pointer_snapshot(runner)
        row = {"update": update + 1, "counts": actual["counts"], "input_tokens": actual["input_tokens"],
            "documents": actual["documents"], "local_keys": keys,
            "physical_microbatches_global": actual["microbatches"], "loss_sums": actual["loss_sums"],
            "counts_match": count_match, "losses_match": loss_match, "gradients": gradient_check,
            "parameters": parameter_check, "adam": adam_check, "scheduler_equal": scheduler_match,
            "pointers_stable": pointer_match, "counters": step["counters"],
            "passed": preparation_passed and count_match and loss_match and gradient_check["passed"]
                and parameter_check["passed"] and adam_check["passed"] and scheduler_match and pointer_match}
        rows.append(row)
        if publish is not None:
            publish(row)
    state = {"model": parameter_snapshot(model), "optimizer": optimizer_snapshot(model, optim),
             "scheduler": schedule.state_dict(), "counters": asdict(counters)}
    return {"arm": arm, "world_size": world_size, "rank": rank, "case": case,
            "preparation_passed": preparation_passed, "updates": rows,
            "passed": all(row["passed"] for row in rows)}, state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--case", choices=("eager", "graph"), required=True)
    parser.add_argument("--arms", default="B,NFR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.arms = tuple(args.arms.split(","))
    if not args.arms or len(set(args.arms)) != len(args.arms) or any(a not in ARMS for a in args.arms):
        parser.error("Use distinct campaign arms")
    if args.device == "cpu" and args.case == "graph":
        parser.error("CUDA graphs have no CPU fallback")
    return args


def main(argv=None):
    args = parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size not in (1, 2):
        raise RuntimeError("Use torchrun with one or two ranks for this bounded acceptance")
    torch.set_num_threads(1)
    runtime = {"device": "cpu", "qualification": "explicit CPU/Gloo diagnostic only"}
    if args.device == "cuda":
        if any(os.environ.get(k) != "0" for k in ("NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING")):
            raise RuntimeError("Set both NCCL async-error flags to 0 and use an external timeout")
        if not Path("/.dockerenv").exists():
            raise RuntimeError("CUDA acceptance must run inside the project container")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        runtime = require_container_gpu()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        device = torch.device("cpu")
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo", timeout=timedelta(seconds=300))
    report = {"schema": "olmo-allocation-acceptance-v1", "runtime": runtime,
              "scope": "Tiny FP32 same-precision allocation acceptance; no checkpoint migration or BF16 claim",
              "world_size": world_size, "case": args.case, "arms": []}
    try:
        if dist.get_rank() == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        for arm in args.arms:
            result, state = run_acceptance(arm, device=device, case=args.case,
                publish=lambda row: write_json(args.output_dir / f"{arm}-rank{dist.get_rank()}-progress.json", row))
            rank_results = [None] * world_size
            dist.all_gather_object(rank_results, result)
            if dist.get_rank() == 0:
                torch.save(state, args.output_dir / f"{arm}-state.pt")
                report["arms"].append({"arm": arm, "ranks": rank_results,
                                      "passed": all(r["passed"] for r in rank_results)})
                write_json(args.output_dir / "report.json", report)
        report["passed"] = all(a["passed"] for a in report["arms"])
        if dist.get_rank() == 0:
            write_json(args.output_dir / "report.json", report)
            print({"passed": report["passed"], "output_dir": str(args.output_dir)}, flush=True)
            if not report["passed"]:
                raise RuntimeError("Allocation acceptance failed; retained evidence must be reviewed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
