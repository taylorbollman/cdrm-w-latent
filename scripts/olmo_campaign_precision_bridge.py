#!/usr/bin/env python3
"""Six fixed-state backwards separating precision from joint backend changes.

No optimizer, DDP, graph capture or numerical acceptance threshold. Model and
loss arithmetic are reused from the component diagnostic, not reimplemented.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_fp32_localize import configure_full_fp32
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_components import (
    RUNTIME_FLAGS, TERMS, component_backward, record_gradients,
    source_hashes as component_sources,
)
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

PATHS = ("fp32_math_eager", "bf16_math_eager", "bf16_flash_triton")
OBJECTIVES = ("ce", "combined")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def configure_path(model, original, path):
    """Restore production flags before each path, so FP32 cannot leak into BF16."""
    if path not in PATHS or set(original) != set(RUNTIME_FLAGS):
        raise ValueError("Unknown path or incomplete production runtime flags")
    if original["attention_precision"] != "mixed":
        raise ValueError("Bridge requires a mixed-attention production baseline")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Bridge requires FP32 master parameters")
    base = model.backbone.backbone
    for name, value in original.items():
        setattr(base, name, value)
    if path == "fp32_math_eager":
        configure_full_fp32(model)
    elif path == "bf16_math_eager":
        base.tile_backend = base.backward_tile_backend = "eager"
    return {"runtime_flags": {key: getattr(base, key) for key in RUNTIME_FLAGS},
            "precision": "fp32" if path.startswith("fp32") else "bf16_mixed",
            "ordinary_sdpa": "FLASH_ATTENTION" if path == PATHS[-1] else "MATH",
            "autocast_cache_enabled": False, "tf32": False}


@contextmanager
def capture_passes(model, fixtures):
    """Observe FBT output once per batch, outside checkpointed block internals."""
    batches = [batch for rank_batches, _ in fixtures for batch in rank_batches]
    records = []

    def capture(module, args, kwargs, output):
        index = len(records)
        if index >= len(batches):
            raise AssertionError("Unexpected repeated FBT forward during diagnostic")
        states = output.pass_hidden_states
        record = {"key": f"fixture-record-{index}",
                  "batch": {key: value.detach().cpu().clone() for key, value in vars(batches[index]).items()},
                  "token_embeddings": kwargs["inputs_embeds"].detach().cpu().clone(),
                  "pass_hidden_states": tuple(h.detach().cpu().clone() for h in states),
                  "total_incoming_cotangents": [None] * len(states)}
        records.append(record)
        for i, h in enumerate(states):
            def save(gradient, i=i, record=record):
                if record["total_incoming_cotangents"][i] is not None:
                    raise AssertionError("Repeated incoming cotangent observation")
                record["total_incoming_cotangents"][i] = gradient.detach().cpu().clone()
            h.register_hook(save)

    handle = model.backbone.register_forward_hook(capture, with_kwargs=True)
    try:
        yield records
    finally:
        handle.remove()
    if len(records) != len(batches):
        raise AssertionError("Missing FBT forward observation")


def tensor_geometry(actual, reference=None):
    a = actual.detach().double().reshape(-1)
    if not a.numel():
        raise ValueError("Geometry requires selected elements")
    row = {"elements": a.numel(), "dtype": str(actual.dtype),
           "finite": bool(torch.isfinite(a).all()), "norm": float(a.norm()),
           "rms": float(a.square().mean().sqrt()), "max_abs": float(a.abs().max())}
    if reference is not None:
        if actual.shape != reference.shape:
            raise ValueError("Geometry shapes differ")
        b = reference.detach().double().reshape(-1)
        an, bn, err = float(a.norm()), float(b.norm()), float((a-b).norm())
        cosine = max(-1., min(1., float(a.dot(b))/(an*bn))) if an and bn else None
        row.update(reference_dtype=str(reference.dtype), reference_norm=bn,
                   difference_norm=err, relative_l2=err/max(bn, 1e-30),
                   cosine=cosine, norm_ratio=an/bn if bn else None,
                   reference_is_zero=bn == 0, max_abs_difference=float((a-b).abs().max()))
    return row


def forward_geometry(records, reference=None):
    """Compare valid tokens only; completely dummy records are explicitly skipped."""
    if reference is not None and len(records) != len(reference):
        raise ValueError("Forward record counts differ")
    result = []
    for i, rec in enumerate(records):
        valid = rec["batch"]["valid_mask"].bool()
        want = None if reference is None else reference[i]
        if want is not None and (tree_digests(rec["batch"]) != tree_digests(want["batch"])):
            raise ValueError("Forward comparison uses different input masks/tokens")
        if not bool(valid.any()):
            result.append({"record": i, "valid_tokens": 0, "skipped_dummy": True})
            continue
        if want is not None and len(rec["pass_hidden_states"]) != len(want["pass_hidden_states"]):
            raise ValueError("Forward pass counts differ")
        for k, h in enumerate(rec["pass_hidden_states"]):
            cot = rec["total_incoming_cotangents"][k]
            if cot is None:
                raise AssertionError("Missing incoming hidden cotangent")
            result.append({"record": i, "pass": k, "valid_tokens": int(valid.sum()),
                "hidden": tensor_geometry(h[valid], None if want is None else want["pass_hidden_states"][k][valid]),
                "total_incoming_cotangent": tensor_geometry(cot[valid], None if want is None else want["total_incoming_cotangents"][k][valid])})
    return result


def source_hashes():
    result = component_sources()
    for name in ("scripts/olmo_campaign_precision_bridge.py", "scripts/olmo_campaign_aux_cotangents.py",
                 "scripts/olmo_f2_graph_backend_probe.py", "docs/reports/olmo-precision-localization/protocol.md"):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def main(argv=None):
    args = parse_args(argv)
    deterministic = configure_determinism(True)  # Must precede any CUDA context.
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Bridge uses one process; no DDP")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-campaign-precision-bridge-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "sources": source_hashes(), "determinism": deterministic, "rows": [],
        "paths": PATHS, "objectives": OBJECTIVES, "optimizer_updates": 0,
        "gradient_cases": 6, "physical_backwards_per_case": 2,
        "fixture": "Original isolated-v1 NFR K4 T16 B2 two virtual rank fixtures, update0",
        "scope": "Six sparse backwards; fixed weights/data/noise. No optimizer/DDP/graphs, quality or performance claim",
        "qualification": "All cross-precision/backend errors descriptive; previous numerical qualifications remain open",
        "cotangent_scope": "TOTAL incoming gradients at pass outputs, including later FBT feedback, not just local losses",
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for relative in report["sources"]:
        dst = args.output_dir/"source-snapshot"/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, dst)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-precision-localization", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started = time.monotonic()

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    failure = None
    try:
        tracker.start({k: report[k] for k in ("scope", "paths", "objectives", "determinism", "qualification")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("loading_source")
        model, recipe, checkpoint, ids, eos = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NFR", torch.device("cuda"))
        original = {key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS}
        fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
            length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
        input_pins = tree_digests([{"batches": [vars(b) for b in batches], "noise": noises} for batches, noises in fixtures])
        report.update(source_checkpoint=checkpoint, recipe=recipe.to_dict(), recipe_sha256=recipe.sha256,
                      fixture_inputs=input_pins, production_runtime_flags=original)
        weights, rng = tree_digests(dict(model.named_parameters())), rng_snapshot()
        for objective in OBJECTIVES:
            references, forward_refs = {}, {}
            for path in PATHS:
                execution = configure_path(model, original, path)
                persist(objective+"/"+path+"/backward")
                backend = SDPBackend.FLASH_ATTENTION if path == PATHS[-1] else SDPBackend.MATH
                with capture_passes(model, fixtures) as observed, sdpa_kernel(backend):
                    metrics = component_backward(model, recipe, fixtures,
                        precision=execution["precision"], layout="sparse", objective=objective)
                grad, snapshot = record_gradients(model, references.get(PATHS[0]),
                    save_cpu=path != PATHS[-1], scope="Descriptive versus FP32 math/eager, no new numerical gate")
                geometry = forward_geometry(observed, forward_refs.get(PATHS[0]))
                finite_forward = all(x["hidden"]["finite"] and x["total_incoming_cotangent"]["finite"]
                    for x in geometry if not x.get("skipped_dummy"))
                finite_losses = all(math.isfinite(v) for v in (metrics["objective"], *metrics["loss_sums"].values()))
                row = {"objective": objective, "path": path, "execution": execution, "metrics": metrics,
                    "normalized_loss_means": {term: metrics["loss_sums"][term]/metrics["counts"][term] for term in TERMS},
                    "gradients_vs_fp32" if path != PATHS[0] else "gradients": grad,
                    "forward_vs_fp32": geometry, "rng_unchanged": rng_unchanged(rng),
                    "forward_fingerprints": tree_digests([{k: r[k] for k in ("batch", "token_embeddings", "pass_hidden_states")} for r in observed]),
                    "memory": memory(), "passed": grad["finite"] and grad["participation_intact"] and finite_forward and finite_losses and rng_unchanged(rng)}
                if objective == "combined":
                    ce_record = next(r for r in report["rows"] if r["path"] == path and r["objective"] == "ce")
                    row["forward_matches_ce_case_exactly"] = row["forward_fingerprints"] == ce_record["forward_fingerprints"]
                    row["passed"] &= row["forward_matches_ce_case_exactly"]
                if path == PATHS[-1]:
                    versus, _ = record_gradients(model, references[PATHS[1]],
                        scope="Descriptive BF16 Flash/Triton versus BF16 math/eager, no numerical gate")
                    row.update(gradients_vs_bf16_math=versus, forward_vs_bf16_math=forward_geometry(observed, forward_refs[PATHS[1]]))
                    if objective == "ce":
                        from scripts.olmo_campaign_aux_cotangents import write_fixture
                        report["auxiliary_fixture"] = write_fixture(args.output_dir/"auxiliary-fixture.json", model,
                            [{k: r[k] for k in ("key", "batch", "token_embeddings", "pass_hidden_states")} for r in observed],
                            {"source_checkpoint": checkpoint, "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256,
                             "execution": execution, "fixture": report["fixture"], "bridge_sources": report["sources"]})
                else:
                    references[path], forward_refs[path] = snapshot, observed
                report["rows"].append(row)
                persist(objective+"/"+path)
                tracker.log(scalar_metrics(row, "diagnostic/"+objective+"/"+path), step=len(report["rows"]))
                print({"stage": report["stage"], "operational_passed": row["passed"], "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
                if not row["passed"]:
                    raise AssertionError("Bridge health/participation/RNG guard failed")
            del references, forward_refs, snapshot, observed
            model.zero_grad(set_to_none=True)
            gc.collect()
        report["integrity"] = {"weights_unchanged": weights == tree_digests(dict(model.named_parameters())),
            "sources_unchanged": source_hashes() == report["sources"], "rng_unchanged": rng_unchanged(rng),
            "inputs_unchanged": input_pins == tree_digests([{"batches": [vars(b) for b in batches], "noise": noises} for batches, noises in fixtures])}
        if not all(report["integrity"].values()):
            raise AssertionError("Fixed diagnostic state changed")
        report.update(status="passed_operational_diagnostic", passed=True)
        persist("complete")
    except BaseException as error:
        failure = error
        report.update(status="failed", passed=False,
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist(report.get("stage", "setup"))
        try:
            tracker.finish(succeeded=report["passed"])
        except BaseException as error:
            report.update(status="failed", passed=False, tracking_finish_error={"type": type(error).__name__})
            if failure is None:
                raise
        finally:
            persist(report.get("stage", "setup"))


if __name__ == "__main__":
    main()
