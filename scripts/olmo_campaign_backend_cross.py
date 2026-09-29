#!/usr/bin/env python3
"""One CE-only crossed backend, with two exactly matched recomputed anchors.

BF16 Flash-SDPA/eager-native-RT is the sole new condition. Math/eager and
Flash/Triton CE anchors are recomputed because prior full gradient vectors were
not retained. Three aggregate cases, six physical-batch backwards, no optimizer,
DDP, graph capture, precision promotion, fixture export or training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
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
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_bridge import (
    capture_passes, configure_path, forward_geometry, source_hashes as bridge_sources,
)
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, TERMS, component_backward, record_gradients
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

MATH = "bf16_math_eager"
TRITON = "bf16_flash_triton"
CROSS = "bf16_flash_eager"
PATHS = (MATH, TRITON, CROSS)
GRADIENT_SUMMARY_KEYS = ("finite", "groups", "originally_missing_zero_materialized", "participation_intact")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-report", type=Path, required=True)
    p.add_argument("--reference-sha256", required=True)
    p.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if len(args.reference_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.reference_sha256):
        p.error("Reference SHA256 must be 64 lowercase hexadecimal characters")
    return args


def configure_cross_path(model, original, path):
    if path not in PATHS:
        raise ValueError("Unsupported crossed-backend path")
    execution = configure_path(model, original, MATH if path == CROSS else path)
    execution["ordinary_sdpa"] = "MATH" if path == MATH else "FLASH_ATTENTION"
    return execution


def source_hashes():
    sources = bridge_sources()
    for name in ("scripts/olmo_campaign_backend_cross.py",
                 "docs/reports/olmo-precision-localization/adaptive-protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def load_reference(path, expected_sha, sources):
    path = Path(path)
    if path.stat().st_size > 64*1024*1024 or sha256_file(path) != expected_sha:
        raise ValueError("Reference report bytes differ from the bounded SHA256 pin")
    report = json.loads(path.read_text())
    if sha256_file(path) != expected_sha:
        raise ValueError("Reference report changed while reading")
    if (report.get("schema") != "olmo-campaign-precision-bridge-v1"
            or report.get("status") != "passed_operational_diagnostic" or report.get("passed") is not True):
        raise ValueError("Reference must be a completed operational bridge report")
    if not report.get("sources") or any(sources.get(name) != sha for name, sha in report["sources"].items()):
        raise ValueError("Reference bridge source pins differ from current sources")
    if (report.get("determinism", {}).get("deterministic_algorithms") is not True
            or not report.get("integrity") or not all(report["integrity"].values())):
        raise ValueError("Reference deterministic/fixed-state integrity is not established")
    for path_name in (MATH, TRITON):
        rows = [r for r in report["rows"] if r.get("objective") == "ce" and r.get("path") == path_name]
        if len(rows) != 1 or not rows[0].get("passed"):
            raise ValueError("Reference must contain exactly one successful CE anchor per backend")
    return report


def reference_anchor(report, path):
    if path not in (MATH, TRITON):
        raise ValueError("The new crossed condition has no previous anchor")
    return next(row for row in report["rows"] if row["objective"] == "ce" and row["path"] == path)


def anchor_comparison(metrics, fingerprints, gradients, previous):
    old_gradients = previous["gradients_vs_fp32"]
    checks = {"metrics_exact": metrics == previous["metrics"],
              "forward_fingerprints_exact": fingerprints == previous["forward_fingerprints"],
              "gradient_group_summaries_exact": all(gradients[key] == old_gradients[key] for key in GRADIENT_SUMMARY_KEYS)}
    return {"passed": all(checks.values()), "checks": checks,
            "scope": "Matched recomputed prior CE anchor; original full gradient vectors were not retained"}


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Backend cross requires one process without DDP")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-campaign-backend-cross-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "sources": source_hashes(), "determinism": determinism, "rows": [], "paths": PATHS,
        "reference_report": str(args.reference_report), "reference_sha256": args.reference_sha256,
        "objective": "ce", "aggregate_backwards": 3, "physical_batch_backwards": 6,
        "new_condition": CROSS, "optimizer_updates": 0,
        "scope": __doc__, "qualification": "Numerical differences descriptive; prior qualifications remain open",
        "cotangent_scope": "TOTAL incoming pass-output gradients, including later feedback; no fixed-cotangent claim",
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for name in report["sources"]:
        dst = args.output_dir/"source-snapshot"/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dst)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-precision-localization", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    try:
        tracker.start({key: report[key] for key in ("scope", "paths", "objective", "new_condition",
                                                   "determinism", "reference_sha256", "qualification")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("verify_reference_and_load_source")
        previous = load_reference(args.reference_report, args.reference_sha256, report["sources"])
        model, recipe, checkpoint, ids, eos = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NFR", torch.device("cuda"))
        original = {key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS}
        fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
            length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
        inputs = tree_digests([{"batches": [vars(b) for b in batches], "noise": noises} for batches, noises in fixtures])
        matched = {"runtime": runtime == previous["runtime"],
                   "checkpoint": checkpoint == previous["source_checkpoint"],
                   "recipe": recipe.to_dict() == previous["recipe"] and recipe.sha256 == previous["recipe_sha256"],
                   "fixture_inputs": inputs == previous["fixture_inputs"],
                   "production_runtime_flags": original == previous["production_runtime_flags"],
                   "determinism": determinism == previous["determinism"],
                   "math_reduction": report["math_sdpa_reduced_precision_reduction"] == previous["math_sdpa_reduced_precision_reduction"],
                   "bf16_matmul_reduction": report["bf16_matmul_reduced_precision_reduction"] == previous["bf16_matmul_reduced_precision_reduction"]}
        report.update(reference_contract_checks=matched, source_checkpoint=checkpoint, recipe=recipe.to_dict(),
                      recipe_sha256=recipe.sha256, fixture_inputs=inputs, production_runtime_flags=original)
        if not all(matched.values()):
            raise AssertionError("Crossed-backend state/input/runtime contract differs from bridge")
        weights, rng = tree_digests(dict(model.named_parameters())), rng_snapshot()
        references, observations = {}, {}
        for path in PATHS:
            execution = configure_cross_path(model, original, path)
            persist(path+"/backward")
            backend = SDPBackend.MATH if path == MATH else SDPBackend.FLASH_ATTENTION
            with capture_passes(model, fixtures) as observed, sdpa_kernel(backend):
                metrics = component_backward(model, recipe, fixtures, precision="bf16_mixed", layout="sparse", objective="ce")
            gradients, snapshot = record_gradients(model, references.get(MATH), save_cpu=path != CROSS,
                scope="Descriptive versus matched BF16 math/eager CE reference; no numerical gate")
            fingerprints = tree_digests([{key: row[key] for key in ("batch", "token_embeddings", "pass_hidden_states")} for row in observed])
            geometry = forward_geometry(observed, observations.get(MATH))
            finite = all(row["hidden"]["finite"] and row["total_incoming_cotangent"]["finite"]
                         for row in geometry if not row.get("skipped_dummy"))
            finite_losses = all(math.isfinite(value) for value in
                (metrics["objective"], *metrics["loss_sums"].values()))
            row = {"path": path, "execution": execution, "metrics": metrics, "finite_losses": finite_losses,
                "normalized_loss_means": {term: metrics["loss_sums"][term]/metrics["counts"][term] for term in TERMS},
                "forward_fingerprints": fingerprints,
                "gradients" if path == MATH else "gradients_vs_bf16_math_eager": gradients,
                "forward_vs_bf16_math_eager": geometry, "rng_unchanged": rng_unchanged(rng), "memory": memory(),
                "passed": gradients["finite"] and gradients["participation_intact"] and finite and finite_losses and rng_unchanged(rng)}
            if path == CROSS:
                comparison, _ = record_gradients(model, references[TRITON],
                    scope="Descriptive crossed Flash/eager versus matched BF16 Flash/Triton CE reference; no numerical gate")
                row.update(gradients_vs_bf16_flash_triton=comparison,
                           forward_vs_bf16_flash_triton=forward_geometry(observed, observations[TRITON]))
            else:
                row["previous_anchor"] = anchor_comparison(metrics, fingerprints, gradients, reference_anchor(previous, path))
                row["passed"] &= row["previous_anchor"]["passed"]
                references[path], observations[path] = snapshot, observed
            report["rows"].append(row)
            persist(path)
            tracker.log(scalar_metrics(row, "diagnostic/"+path), step=len(report["rows"]))
            print({"stage": path, "operational_passed": row["passed"], "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
            if not row["passed"]:
                raise AssertionError("Crossed-backend anchor, finite/participation or RNG guard failed")
        del references, observations, snapshot, observed
        model.zero_grad(set_to_none=True)
        gc.collect()
        report["integrity"] = {"weights_unchanged": weights == tree_digests(dict(model.named_parameters())),
            "sources_unchanged": source_hashes() == report["sources"], "rng_unchanged": rng_unchanged(rng),
            "reference_unchanged": sha256_file(args.reference_report) == args.reference_sha256,
            "inputs_unchanged": inputs == tree_digests([{"batches": [vars(b) for b in batches], "noise": noises} for batches, noises in fixtures])}
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
