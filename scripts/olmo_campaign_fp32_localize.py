#!/usr/bin/env python3
"""Fixed-state FP32 sparse/prepared objective localization on actual OLMo weights.

This is mathematical-path agreement at FP32, not FP32-versus-BF16 agreement,
and cannot establish harmlessness of the separately observed BF16 discrepancy.
No DDP, CUDA graph, optimizer update, or performance measurement is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from scripts.olmo_campaign_ddp_probe import (
    canonical_backward, construct, fixture_for_update, prepared_backward,
    source_hashes as campaign_sources,
)
from scripts.olmo_campaign_graph_probe import compare_metrics, rng_snapshot, rng_unchanged
from scripts.olmo_campaign_probe import gradient_record, memory
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def configure_full_fp32(model):
    """Change only this model's backend flags before constructing any layout."""
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("FP32 diagnostic requires existing FP32 master parameters")
    base = model.backbone.backbone
    base.attention_backend = "sdpa"
    base.ordinary_attention_backend = "sdpa"
    base.attention_precision = "fp32"
    base.tile_backend = base.backward_tile_backend = "eager"
    base.ordinary_pointwise_backend = "eager"
    base.ordinary_rope_backend = "native"
    names = ("attention_backend", "ordinary_attention_backend", "attention_precision",
             "tile_backend", "backward_tile_backend", "backward_memory", "rt_implementation",
             "ordinary_pointwise_backend", "ordinary_rope_backend", "ordinary_activation_checkpointing",
             "cast_weights_once", "reuse_rope", "kv_only_writes")
    return {"precision": "full FP32; autocast disabled; FP32 parameters and raw gradients",
            "ordinary_sdpa_dispatch": "forced MATH",
            "tf32": False, "float32_matmul_precision": "highest",
            "ddp": False, "cuda_graphs": False, "optimizer_updates": 0,
            "runtime_flags": {name: getattr(base, name) for name in names}}


def acceptance(metrics, gradient, *, weights_unchanged, rng_preserved):
    """Use the same budgets as the preceding BF16 sparse/prepared diagnostic."""
    comparison = gradient.get("comparison", {})
    return {"passed": bool(metrics.get("passed") and gradient.get("finite")
                            and comparison.get("all_parameters_close")
                            and weights_unchanged and rng_preserved),
            "losses_pass": bool(metrics.get("passed")),
            "raw_gradients_pass": bool(gradient.get("finite") and comparison.get("all_parameters_close")),
            "weights_unchanged": bool(weights_unchanged), "rng_unchanged": bool(rng_preserved),
            "scope": "FP32 sparse/prepared mathematical-path agreement only; BF16 qualification remains open"}


def source_hashes():
    sources = campaign_sources()
    sources[str(Path(__file__).relative_to(ROOT))] = sha256_file(__file__)
    return dict(sorted(sources.items()))


def main(argv=None):
    args = parse_args(argv)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("FP32 localization must run without a process group")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-campaign-fp32-localization-v1", "status": "running", "passed": False,
              "runtime": runtime, "sources": source_hashes(), "rows": [],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "arm": "NFR", "length": 16, "physical_batch": 2, "virtual_ranks": 2,
              "fixture_update": 0, "optimizer_updates": 0,
              "budgets": {"raw_gradient_atol": 3e-5, "raw_gradient_rtol": 3e-4,
                          "loss_sum_atol": 1e-5, "loss_sum_rtol": 3e-6,
                          "objective_atol": 1e-6, "objective_rtol": 3e-6},
              "scope": "Same actual pretrained weights, same FP32 runtime, sparse versus prepared dense losses; not independent BF16 acceptance, DDP, graphs, throughput or training quality",
              "checkpoints": "Pinned source reused; CPU raw-gradient reference only; no model checkpoint or optimizer state created"}
    for relative in report["sources"]:
        destination = args.output_dir / "source-snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-campaign-two-gpu-readiness", name="campaign-fp32-" + args.output_dir.name,
        preserve_state=preserve_local_rng)
    started = time.monotonic()

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - started, wandb=tracker.record)
        write_json(args.output_dir / "report.json", report)

    def publish(stage, row):
        report["rows"].append({"stage": stage, **row})
        persist(stage)
        tracker.log(scalar_metrics(row, "diagnostic/" + stage), step=len(report["rows"]))
        print({"stage": stage, "passed": row.get("passed"), "elapsed_seconds": report["elapsed_seconds"]}, flush=True)

    failure = None
    try:
        tracker.start({key: report[key] for key in ("arm", "length", "physical_batch", "virtual_ranks", "scope", "budgets")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("loading_source")
        model, recipe, checkpoint, ids, eos = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NFR", torch.device("cuda"))
        report.update(recipe=recipe.to_dict(), recipe_sha256=recipe.sha256, source_checkpoint=checkpoint,
                      execution=configure_full_fp32(model))
        fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
                       length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
        report["fixture_inputs"] = tree_digests([
            {"batches": [vars(batch) for batch in batches], "noise": noise}
            for batches, noise in fixtures])
        weights = tree_digests(dict(model.named_parameters()))
        before = rng_snapshot()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            persist("canonical_sparse_backward")
            canonical = canonical_backward(model, recipe, fixtures, precision="fp32")
            canonical_gradients, reference = gradient_record(model, save_cpu=True)
            publish("canonical_sparse", {"metrics": canonical, "gradients": canonical_gradients,
                "rng_unchanged": rng_unchanged(before), "memory": memory(),
                "passed": canonical_gradients["finite"] and rng_unchanged(before)})
            if not report["rows"][-1]["passed"]:
                raise AssertionError("Canonical FP32 reference is not finite or changed RNG")
            persist("prepared_dense_backward")
            prepared = prepared_backward(model, recipe, fixtures, precision="fp32")
            gradients, _ = gradient_record(model, reference)
            metrics = compare_metrics(prepared, canonical)
        state_exact = tree_digests(dict(model.named_parameters())) == weights
        result = acceptance(metrics, gradients, weights_unchanged=state_exact, rng_preserved=rng_unchanged(before))
        publish("prepared_dense_vs_sparse", {**result, "metrics": prepared,
            "metric_comparison": metrics, "gradients": gradients, "memory": memory()})
        if not result["passed"]:
            raise AssertionError("FP32 sparse/prepared path comparison failed unchanged diagnostic budgets")
        if source_hashes() != report["sources"]:
            raise AssertionError("Runtime sources changed during FP32 localization")
        model.zero_grad(set_to_none=True)
        del reference, model
        report.update(status="passed", passed=True)
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
