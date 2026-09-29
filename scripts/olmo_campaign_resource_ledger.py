#!/usr/bin/env python3
"""CPU-only parameter and matrix-work ledger for all eight campaign arms.

No checkpoint loading, model forward, optimizer update, CUDA or timing claims.
The default workload is the retained packed first update: T1024, two ranks,
B12/rank and 22 physical slots/rank, including dummy rows. Each arm is an
accounting counterfactual at this SAME footprint, not a measured operating point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_recipe import ARMS, CampaignRecipe, build_campaign_adamw, build_campaign_model
from cdrm.pretrained.document_policy import CONTINUOUS_STREAM
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.resource_estimates import (
    architecture_parameter_counts, estimate_training_resources, parameter_inventory,
)

REFERENCE = ROOT/".runtime/olmo-packed-campaign/pretrained-write-02/report.json"
REFERENCE_SHA = "a3c1b48913ecc95f087cc1a79ddc3c70e6ac0642e0f0e5ee0f3883b42b230754"
TERMS = ("ce", "latent", "kl", "predictor")
LOSS_COMPONENTS = ("ce_readout_forward_recompute_backward", "nextlat_predictor_forward_backward",
                   "kl_readout_forward_recompute_backward")


def nonnegative(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def selected_work(batch, *, nextlat, document_policy):
    """Exact sparse selections, including the latent/KL predictor-source union."""
    masks = build_nextlat_masks(batch, document_policy=document_policy)
    latent = masks["latent"] if nextlat else torch.zeros_like(masks["latent"])
    kl = masks["kl"] if nextlat else torch.zeros_like(masks["kl"])
    kl_source = torch.nn.functional.pad(kl, (0, 1)) if batch.input_ids.shape[1] > 1 else latent
    return {"ce": int(masks["ce"].sum()), "latent": int(latent.sum()),
            "kl": int(kl.sum()), "predictor": int((latent | kl_source).sum())}


def executed_work(physical_rows, length, selected, *, nextlat, layout):
    """Loss counts describe either selected sparse positions or dense capacity."""
    for name, value in (("physical_rows", physical_rows), ("length", length)):
        nonnegative(name, value)
    if physical_rows < 1 or length < 3 or layout not in ("sparse", "dynamic_dense"):
        raise ValueError("Require positive rows, T>=3 and an explicit loss layout")
    if set(selected) != set(TERMS):
        raise ValueError("Selected work needs CE, latent, KL and predictor-union counts")
    for name, value in selected.items():
        nonnegative(name, value)
    pairs, triples = physical_rows*(length-1), physical_rows*(length-2)
    if (max(selected["ce"], selected["latent"], selected["predictor"]) > pairs
            or selected["kl"] > triples
            or not max(selected["latent"], selected["kl"]) <= selected["predictor"] <= selected["latent"]+selected["kl"]):
        raise ValueError("Selected work exceeds capacity or violates predictor-source union")
    if not nextlat and any(selected[t] for t in ("latent", "kl", "predictor")):
        raise ValueError("Disabled NextLat cannot have auxiliary selections")
    if layout == "sparse":
        return dict(selected)
    return {"ce": pairs, "latent": pairs if nextlat else 0,
            "kl": triples if nextlat else 0, "predictor": pairs if nextlat else 0}


def parameter_card(model, recipe, optimizer):
    """Count actual module/optimizer ownership, deduplicating tied weights."""
    named = dict(model.named_parameters())
    active = {n for n in named if not n.startswith("backbone.fusion.") or recipe.feedback}
    deployable = {n for n in active if not n.startswith("predictor.")}
    observed = parameter_inventory(model, optimizer=optimizer, executed_names=active, inference_names=deployable)
    architecture = architecture_parameter_counts(model.backbone.config, fbt=recipe.feedback,
        nextlat=model.config if recipe.nextlat else None)
    groups = {}
    for name in ("backbone", "fusion", "predictor"):
        def group(n):
            return "predictor" if n.startswith("predictor.") else "fusion" if n.startswith("backbone.fusion.") else "backbone"
        chosen = {n:p for n,p in named.items() if group(n) == name}
        owned = {id(p) for g in optimizer.param_groups for p in g["params"]}
        groups[name] = {"resident_parameters": sum(p.numel() for p in chosen.values()),
            "trainable_parameters": sum(p.numel() for p in chosen.values() if p.requires_grad),
            "optimizer_owned_parameters": sum(p.numel() for p in chosen.values() if id(p) in owned),
            "branch_active_parameters": sum(p.numel() for n,p in chosen.items() if n in active),
            "deployable_parameters": sum(p.numel() for n,p in chosen.items() if n in deployable),
            "parameter_tensors": len(chosen)}
    checks = {"tied_readout_identity": model.backbone.readout_weight is model.backbone.token_embeddings.weight,
        "active_matches_architecture": observed["executed_declared"] == architecture["training_architecture"],
        "deployable_matches_architecture": observed["deployable_inference_declared"] == architecture["deployable_inference"],
        "optimizer_matches_trainable": observed["optimizer_owned"] == observed["trainable"],
        "active_matches_trainable": observed["executed_declared"] == observed["trainable"],
        "no_optimizer_state": not optimizer.state,
        "no_gradients": observed["gradient_participating"] == 0}
    if not all(checks.values()):
        raise AssertionError("Campaign parameter ownership differs from its architecture")
    return {"observed_inventory": observed, "groups": groups, "architecture": architecture, "checks": checks,
        "scope": "Actual CPU modules and fresh optimizer ownership; no loaded weights, execution or measured device memory"}


def matrix_card(config, recipe, nextlat_config, physical_rows, selected, *, layout):
    """Sum matrix arithmetic over scheduled physical rows, not useful tokens.

Existing formulas are linear in B. Aggregating B over slots is arithmetic only,
not concatenating inputs or predicting the kernel's batch-dependent behavior.
Backbone work assumes every scheduled slot executes even when masks are empty.
"""
    work = executed_work(physical_rows, recipe.sequence_length, selected, nextlat=recipe.nextlat, layout=layout)
    ncfg = nextlat_config if recipe.nextlat else None
    estimate = estimate_training_resources(config, batch_size=physical_rows,
        sequence_length=recipe.sequence_length, mode=recipe.mode(), nextlat=ncfg,
        ordinary_checkpointing=True, backward_memory="recompute", kv_only_writes=True)
    components = [dict(name=c.name, minimum=c.minimum, maximum=c.maximum, description=c.description)
                  for c in estimate.components if c.name not in LOSS_COMPONENTS]
    passes, d, v = recipe.mode().num_passes, config.model_dim, config.vocab_size
    # Current chunked losses: CE forward/replay/dHidden/dReadout; KL teacher+
    # student forward, both replay, student dHidden. Teacher/readout are detached.
    loss = {LOSS_COMPONENTS[0]: passes*work["ce"]*8*d*v}
    if recipe.nextlat:
        p = nextlat_config.hidden_dim
        loss[LOSS_COMPONENTS[1]] = passes*work["predictor"]*6*(3*d*p+p*p)
        loss[LOSS_COMPONENTS[2]] = passes*work["kl"]*10*d*v
    components.extend({"name":name, "minimum":count, "maximum":count,
        "description":"Current loss matrix arithmetic at this layout's executed position count"} for name,count in loss.items())
    return {"layout":layout, "executed_positions_once_per_pass":work,
        "executed_positions_across_passes":{k:passes*v for k,v in work.items()},
        "matrix_flops_minimum":sum(c["minimum"] for c in components),
        "matrix_flops_maximum":sum(c["maximum"] for c in components), "components":components,
        "ordinary_block_calls_per_physical_slot":estimate.ordinary_block_calls_per_microbatch,
        "rt_block_calls_per_physical_slot":estimate.rt_block_calls_per_microbatch,
        "assumptions":["All active weights trainable; this is NOT frozen-backbone warmup accounting.",
          "Padded/dummy rows execute dense backbone work. Sparse is a selected-loss accounting counterfactual, not another measured trainer.",
          "Positive pass coefficients weight objectives but do not reduce the number of executed passes.",
          "Multiply-add=2 matrix FLOPs; ordinary causal/full-square and checkpoint early-stop bounds are accounting conventions, not measured hardware instructions.",
          "Excluded: norms, RoPE, activations, softmax/CE/KL pointwise, gathers/masks, casts, optimizer/clipping, communication, launch overhead, hardware padding and compilation/capture warmup.",
          "No throughput, MFU, speed, memory-capacity or comparative-quality claim."]}


def workload_from_report(path=REFERENCE, expected_sha=REFERENCE_SHA):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64*1024**2 or sha256_file(path) != expected_sha:
        raise ValueError("Workload reference differs from its bounded immutable pin")
    report = json.loads(path.read_text())
    if sha256_file(path) != expected_sha:
        raise ValueError("Reference changed while reading")
    if (report.get("status") != "passed" or report.get("phase") != "write"
            or report.get("document_policy") != CONTINUOUS_STREAM or report.get("arm") != "NFR"):
        raise ValueError("Require the completed packed NFR write workload")
    update = report["updates"][0]
    length, batch, slots = report["length"], report["physical_batch_per_rank"], update["microbatches_per_rank"]
    ranks = update["rank_accounting"]
    if len(ranks) != report["configuration"]["world_size"]:
        raise ValueError("Rank accounting and world size differ")
    result = []
    for rank, values in enumerate(ranks):
        for key in ("physical_rows", "packed_rows", "empty_rows", "valid_tokens", "padding_tokens", "ce_targets", "latent_pairs", "kl_triples"):
            nonnegative(key, values[key])
        if (values["physical_rows"] != batch*slots or values["physical_rows"] != values["packed_rows"]+values["empty_rows"]
                or values["valid_tokens"]+values["padding_tokens"] != values["physical_rows"]*length
                or values["ce_targets"] != values["valid_tokens"]-values["packed_rows"]
                or values["latent_pairs"] != values["ce_targets"]-values["excluded_boundary_latent_pairs"]):
            raise ValueError("Packed rank rows/targets/padding accounting differs")
        # All same-document pairs are supervised in this fixed pretraining
        # workload. Every KL-source triple is therefore already a latent pair.
        selected = {"ce":values["ce_targets"], "latent":values["latent_pairs"],
                    "kl":values["kl_triples"], "predictor":values["latent_pairs"]}
        executed_work(values["physical_rows"], length, selected, nextlat=True, layout="sparse")
        result.append({"rank":rank, "physical_batch":batch, "slots":slots, **values, "selected":selected})
    for source,target in (("valid_tokens","valid_tokens"),("packed_rows","packed_rows"),
                          ("ce_targets","ce_targets"),("latent_pairs","latent_pairs"),("kl_triples","kl_triples")):
        if sum(r[source] for r in result) != update["logical_counts"][target]:
            raise ValueError("Rank counts fail to reproduce global logical counts")
    return {"source_report":str(path), "source_sha256":expected_sha, "source_update":update["label"],
        "length":length, "document_policy":CONTINUOUS_STREAM, "world_size":len(ranks), "ranks":result,
        "predictor_union_authority":"All within-document pairs supervised; KL-source triples subset latent pairs in this pinned packed workload"}


def build_card(model, recipe, optimizer, workload):
    ranks = []
    for rank in workload["ranks"]:
        selected = {k:(v if k == "ce" or recipe.nextlat else 0) for k,v in rank["selected"].items()}
        ranks.append({"rank":rank["rank"], "physical_rows":rank["physical_rows"], "physical_slots":rank["slots"],
            "valid_input_tokens":rank["valid_tokens"], "allocated_input_tokens":rank["physical_rows"]*recipe.sequence_length,
            "padding_tokens":rank["padding_tokens"], "dummy_rows":rank["empty_rows"],
            "supervised_positions_once_per_update":selected,
            "sparse":matrix_card(model.backbone.config, recipe, model.config, rank["physical_rows"], selected, layout="sparse"),
            "dynamic_dense":matrix_card(model.backbone.config, recipe, model.config, rank["physical_rows"], selected, layout="dynamic_dense")})
    totals = {key:sum(r[key] for r in ranks) for key in ("physical_rows","physical_slots","valid_input_tokens","allocated_input_tokens","padding_tokens","dummy_rows")}
    totals["supervised_positions_once_per_update"] = {t:sum(r["supervised_positions_once_per_update"][t] for r in ranks) for t in TERMS}
    passes = recipe.mode().num_passes
    totals["allocated_pass_tokens"] = totals["allocated_input_tokens"]*passes
    totals["valid_pass_tokens"] = totals["valid_input_tokens"]*passes
    totals["supervised_positions_across_passes"] = {t:v*passes for t,v in totals["supervised_positions_once_per_update"].items()}
    for layout in ("sparse", "dynamic_dense"):
        totals[layout] = {key:sum(r[layout][key] for r in ranks) for key in ("matrix_flops_minimum","matrix_flops_maximum")}
        totals[layout]["executed_positions_once_per_pass"] = {t:sum(r[layout]["executed_positions_once_per_pass"][t] for r in ranks) for t in TERMS}
        totals[layout]["executed_positions_across_passes"] = {t:v*passes for t,v in totals[layout]["executed_positions_once_per_pass"].items()}
        for kind in ("ordinary", "rt"):
            totals[layout][kind+"_block_invocations"] = sum(r["physical_slots"]*r[layout][kind+"_block_calls_per_physical_slot"] for r in ranks)
    return {"arm":recipe.arm, "recipe":recipe.to_dict(), "parameters":parameter_card(model, recipe, optimizer),
            "ranks":ranks, "totals":totals}


def source_hashes():
    names = ("scripts/olmo_campaign_resource_ledger.py", "tests/test_campaign_resource_ledger.py",
        "docs/reports/olmo-campaign-resource-ledger/protocol.md")
    # All local pretrained implementation files are small; retain a complete
    # direct model/accounting inventory rather than infer transitive imports.
    paths = [ROOT/name for name in names] + sorted((ROOT/"cdrm/pretrained").glob("*.py"))
    return {str(path.relative_to(ROOT)):sha256_file(path) for path in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, default=REFERENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error("Report must remain in the persistent project")
    if torch.cuda.is_initialized():
        raise RuntimeError("Resource ledger must run CPU-only before any CUDA initialization")
    torch.set_num_threads(2)
    workload = workload_from_report(args.reference_report)
    sources = source_hashes()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in sources:
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    report = {"schema":"olmo-campaign-resource-ledger-v1", "status":"running", "started_utc":datetime.now(timezone.utc).isoformat(),
        "scope":__doc__, "sources":sources, "workload":workload, "cards":[],
        "construction":"Actual randomly initialized CPU native modules; no pretrained weight/checkpoint load or forward. AdamW construction only, no state materialization or step.",
        "qualification":"Matrix estimates only; all8 arms share the retained NFR physical footprint for accounting, not a capacity recommendation. Sparse/dense arithmetic differs; numerical equivalence is not claimed."}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20260929)
        backbone = OLMoTiledRTForCausalLM(OLMoConfig.native_1b(), device="cpu", dtype=torch.float32)
        for arm in ARMS:
            recipe = CampaignRecipe(arm=arm, sequence_length=workload["length"], document_policy=CONTINUOUS_STREAM)
            model = build_campaign_model(backbone, recipe)
            optimizer = build_campaign_adamw(model, recipe, fused=False)
            report["cards"].append(build_card(model, recipe, optimizer, workload))
            write_json(args.output_dir/"report.json", report)
            print({"arm":arm, "trainable":report["cards"][-1]["parameters"]["observed_inventory"]["trainable"]}, flush=True)
            del optimizer, model
            gc.collect()
        del backbone
    report["integrity"] = {"sources_unchanged":source_hashes() == sources,
        "reference_unchanged":sha256_file(args.reference_report) == REFERENCE_SHA,
        "cpu_only":not torch.cuda.is_initialized(), "all_eight_arms":len(report["cards"]) == 8}
    if not all(report["integrity"].values()):
        raise AssertionError("CPU resource-ledger integrity failure")
    report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir/"report.json", report)


if __name__ == "__main__":
    main()
