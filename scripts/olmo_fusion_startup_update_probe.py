#!/usr/bin/env python3
"""Three counterfactual fusion Adam steps from one saved startup boundary.

Zero-gradient control, FP32/math, and production BF16/Flash each restore the same
fusion weights, Adam moments and scheduler. Two physical held-out microbatches
form each actual gradient. No training cursor is advanced or checkpoint replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from cdrm.pretrained.lm_training import _rng_state, optimizer_ownership
from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct
from scripts.olmo_campaign_precision_bridge import configure_path, tensor_geometry
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, fixture_pins
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_long_probe import load_long_fixture, source_hashes as long_sources
from scripts.olmo_fusion_startup_train import (
    FUSION_NAMES, TRAINING, _checkpoint_payload, assert_fusion_only, build_optimizer,
    ce_loss_sums, freeze_for_startup, frozen_state_pins, load_fusion_checkpoint,
    source_hashes as training_sources,
)
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

ZERO = "zero_gradient_adam_control"
CASES = (ZERO, FP32, BF16)


def source_hashes():
    sources = long_sources()
    for name in ("scripts/olmo_fusion_startup_update_probe.py", "tests/test_fusion_startup_update_probe.py",
                 "docs/reports/olmo-fusion-startup/update-probe-protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def fusion_values(model, *, gradients=False):
    parameters = dict(model.named_parameters())
    result = {}
    for name in FUSION_NAMES:
        tensor = parameters[name].grad if gradients else parameters[name]
        if tensor is None or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError("Fusion tensor missing, non-FP32 or nonfinite")
        result[name] = tensor.detach().cpu().clone()
    return result


def moment_values(model, optimizer):
    result, steps = {"exp_avg": {}, "exp_avg_sq": {}}, {}
    parameters = dict(model.named_parameters())
    for name in FUSION_NAMES:
        state = optimizer.state[parameters[name]]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Require populated ordinary Adam moments for both fusion matrices")
        steps[name] = float(state["step"])
        for key in result:
            tensor = state[key]
            if tensor.dtype != torch.float32 or tensor.shape != parameters[name].shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Saved Adam moment shape/dtype/finiteness differs")
            result[key][name] = tensor.detach().cpu().clone()
    return result, steps


def geometry(actual, reference=None):
    if reference is not None and actual.keys() != reference.keys():
        raise ValueError("Compared fusion tensor inventories differ")
    rows = {name: tensor_geometry(value, None if reference is None else reference[name])
            for name, value in actual.items()}
    sums = {"actual": 0., "reference": 0., "error": 0., "dot": 0.}
    elements = 0
    for name, tensor in actual.items():
        value = tensor.double()
        sums["actual"] += float(value.square().sum()); elements += value.numel()
        if reference is not None:
            other = reference[name].double()
            sums["reference"] += float(other.square().sum())
            sums["error"] += float((value-other).square().sum())
            sums["dot"] += float((value*other).sum())
    a, b = math.sqrt(sums["actual"]), math.sqrt(sums["reference"])
    combined = {"elements": elements, "norm": a, "finite": all(row["finite"] for row in rows.values())}
    if reference is not None:
        error = math.sqrt(sums["error"])
        combined.update(reference_norm=b, difference_norm=error,
            relative_l2=error/b if b else None, norm_ratio=a/b if b else None,
            cosine=max(-1., min(1., sums["dot"]/(a*b))) if a and b else None,
            reference_is_zero=b == 0)
    return {"all": combined, "tensors": rows}


def difference(actual, reference):
    if actual.keys() != reference.keys():
        raise ValueError("Difference tensor inventories differ")
    # Subtract exact stored FP32 values in FP64, so tiny master-weight updates
    # are not lost by further subtraction roundoff in observation code.
    return {name: actual[name].double()-reference[name].double() for name in actual}


def decay_delta(model, optimizer):
    result = {}
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    with torch.no_grad():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                original = parameter.detach().cpu().clone()
                decayed = parameter.detach().clone().mul_(1-group["lr"]*group["weight_decay"]).cpu()
                result[names[id(parameter)]] = decayed.double()-original.double()
    if set(result) != set(FUSION_NAMES):
        raise ValueError("Decay observation must own exactly the fusion parameters")
    return result


def ce_gradients(model, recipe, fixtures, *, path, original_flags):
    """Frozen-backbone CE, with identical forward/recompute dispatch scope."""
    assert_fusion_only(model)
    if path not in (FP32, BF16) or len(fixtures) != 2:
        raise ValueError("Require two held-out records and a declared precision")
    batches = tuple(batch for records, _ in fixtures for batch in records)
    noises = tuple(noise for _, records in fixtures for noise in records)
    if len(batches) != 2 or len(noises) != 2:
        raise ValueError("Require exactly two physical microbatches")
    count = sum(int(build_nextlat_masks(batch, document_policy="isolated-v1")["ce"].sum()) for batch in batches)
    if count <= 0:
        raise ValueError("No held-out CE targets")
    execution = configure_path(model, original_flags, path)
    device = next(model.parameters()).device
    # CPU is solely an explicit unit-test oracle. The production CLI requires
    # the CUDA container before constructing anything and never falls back.
    backend = SDPBackend.FLASH_ATTENTION if path == BF16 and device.type == "cuda" else SDPBackend.MATH
    model.zero_grad(set_to_none=True)
    metrics = {"objective": 0., "ce_sum": 0., "ce_targets": count, "physical_backwards": 2,
               "pass_ce_sums": [0.]*4, "execution": execution, "sdpa_backend": backend.name}
    for batch, noise in zip(batches, noises):
        with sdpa_kernel(backend), torch.autocast(device.type, enabled=False):
            with torch.autocast(device.type, dtype=torch.bfloat16,
                    enabled=path == BF16, cache_enabled=False):
                result = ce_loss_sums(model, recipe, batch.to(device), tuple(value.to(device) for value in noise))
                objective = result.sums["ce"]/count
            if not bool(torch.isfinite(objective)) or not objective.requires_grad:
                raise FloatingPointError("Held-out CE is nonfinite or detached")
            # Keep forced SDPA active, but match production's backward outside
            # forward autocast. Non-reentrant checkpoints restore their saved
            # forward autocast context themselves during recomputation.
            objective.backward()
        metrics["objective"] += float(objective.detach())
        metrics["ce_sum"] += float(result.sums["ce"].detach())
        for index, loss in enumerate(result.pass_losses):
            metrics["pass_ce_sums"][index] += float(loss.sums["ce"].detach())
    assert_fusion_only(model)
    values = fusion_values(model, gradients=True)
    if any(not bool(value.any()) for value in values.values()):
        raise FloatingPointError("A fusion matrix received zero held-out gradient")
    return metrics


def measure_updates(model, recipe, fixtures, optimizer, scheduler, *, restore, original_flags, publish):
    """Exactly three optimizer calls; all start from the same validated boundary."""
    before_inputs = fixture_pins(fixtures)
    frozen = frozen_state_pins(model)
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    modes = {name: module.training for name, module in model.named_modules()}
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    baseline = initial = common_decay = initial_boundary = None
    snapshots, rows = {}, []
    calls = 0
    result = None
    try:
        for path in CASES:
            model.zero_grad(set_to_none=True)
            restored = restore()
            ownership = optimizer_ownership(model.backbone.fusion, optimizer)
            if ownership != [["state_proj.weight", "token_gate.weight"]]:
                raise ValueError("Update probe optimizer ownership changed")
            parameters = assert_fusion_only(model)
            start = fusion_values(model)
            _, steps_before = moment_values(model, optimizer)
            if any(value != restored["counters"]["optimizer_updates"] for value in steps_before.values()):
                raise ValueError("Saved Adam steps differ from the completed update counter")
            boundary = tree_digests({"weights": model.backbone.fusion.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": _rng_state(None), "counters": restored["counters"], "cursor": restored["data_cursor"]})
            if initial_boundary is None:
                initial_boundary, initial = boundary, start
                common_decay = decay_delta(model, optimizer)
            elif boundary != initial_boundary:
                raise AssertionError("Candidate did not start from the same complete saved boundary")
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            weight_decays = [group["weight_decay"] for group in optimizer.param_groups]
            if path == ZERO:
                for parameter in parameters:
                    parameter.grad = torch.zeros_like(parameter)
                metrics = {"objective": None, "physical_backwards": 0,
                           "meaning": "Zero gradients, not None: moments age and AdamW applies decay"}
            else:
                metrics = ce_gradients(model, recipe, fixtures, path=path, original_flags=original_flags)
            raw = fusion_values(model, gradients=True)
            norm = torch.nn.utils.clip_grad_norm_(parameters, TRAINING["max_grad_norm"],
                                                   error_if_nonfinite=True, foreach=False)
            clipped = fusion_values(model, gradients=True)
            optimizer.step(); calls += 1
            scheduler.step()
            after = fusion_values(model)
            moments, steps = moment_values(model, optimizer)
            if any(steps[name] != steps_before[name]+1 for name in FUSION_NAMES):
                raise AssertionError("Adam must advance each owned step exactly once")
            delta = difference(after, initial)
            if path == ZERO:
                baseline = delta
            snapshot = {"raw_gradient": raw, "clipped_gradient": clipped,
                "actual_master_delta": delta, "delta_without_common_decay": difference(delta, common_decay),
                "delta_beyond_zero_gradient_adam": difference(delta, baseline),
                "exp_avg": moments["exp_avg"], "exp_avg_sq": moments["exp_avg_sq"]}
            summaries = {name: geometry(value) for name,value in snapshot.items()}
            if path == BF16:
                comparisons = {name: geometry(value, snapshots[FP32][name]) for name,value in snapshot.items()}
            else:
                comparisons = None
            row = {"case": path, "metrics": metrics, "lr_used": learning_rates,
                "weight_decay": weight_decays, "saved_steps": steps_before, "candidate_steps": steps,
                "initial_boundary_pins": initial_boundary,
                "lr_next": [group["lr"] for group in optimizer.param_groups],
                "scheduler_after": tree_digests(scheduler.state_dict()),
                "gradient_norm_before_clip": float(norm), "max_grad_norm": TRAINING["max_grad_norm"],
                "clip_scale": min(1., TRAINING["max_grad_norm"]/(float(norm)+1e-6)),
                "same_initial_boundary": True, "actual_optimizer_calls_so_far": calls,
                "observations": summaries, "bf16_vs_fp32": comparisons,
                "tensor_pins": tree_digests(snapshot),
                "finite": all(value["all"]["finite"] for value in summaries.values())}
            if not row["finite"]:
                raise FloatingPointError("Nonfinite candidate update observation")
            rows.append(row); publish(row)
            if path == FP32:
                snapshots[FP32] = snapshot
            model.zero_grad(set_to_none=True)
        if calls != 3:
            raise AssertionError("Three-step budget violated")
        result = {"rows": rows, "optimizer_calls": calls,
            "common_decay": geometry(common_decay), "zero_gradient_adam_delta": geometry(baseline),
            "interpretation": "Actual master deltas include common decay and history. Subtracting zero-gradient Adam removes their shared control, not a linear decomposition of Adam."}
        return result
    finally:
        model.zero_grad(set_to_none=True)
        restored = restore()
        configure_path(model, original_flags, BF16)
        restored_boundary = tree_digests({"weights": model.backbone.fusion.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": _rng_state(None), "counters": restored["counters"], "cursor": restored["data_cursor"]})
        checks = {"frozen_state": frozen_state_pins(model) == frozen,
            "saved_boundary_restored": initial_boundary is None or restored_boundary == initial_boundary,
            "fixtures": fixture_pins(fixtures) == before_inputs,
            "trainability": flags == {n:p.requires_grad for n,p in model.named_parameters()},
            "modes": modes == {n:m.training for n,m in model.named_modules()},
            "identities": identities == {n:id(p) for n,p in model.named_parameters()},
            "gradients_cleared": all(p.grad is None for p in model.parameters())}
        if not all(checks.values()):
            raise AssertionError("Update probe restoration/frozen-state integrity failed")
        if result is not None:
            result["restoration"] = checks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for value in (args.checkpoint_sha256, args.fixture_sha256):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error("Require independent lowercase SHA256 pins")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error("Evidence must remain within the persistent project")
    return args


def main(argv=None):
    args = parse_args(argv)
    deterministic = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Update probe is one GPU/process, without DDP")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    for name in sources:
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    report = {"schema": "olmo-fusion-startup-update-probe-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime, "determinism": deterministic,
        "sources": sources, "checkpoint_sha256": args.checkpoint_sha256, "fixture_sha256": args.fixture_sha256,
        "rows": [], "optimizer_call_budget": 3, "aggregate_backwards": 2, "physical_backwards": 4,
        "qualification": "Counterfactual fusion-only held-out updates from saved Adam; no training continuation, RT, auxiliaries or production BF16 clearance"}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-fusion-startup", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    try:
        tracker.start({key:report[key] for key in ("checkpoint_sha256", "fixture_sha256", "qualification", "determinism")})
        persist("construct")
        model, recipe, source, _, _ = construct(SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NF", torch.device("cuda"))
        original = {name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS}
        freeze_for_startup(model)
        payload = _checkpoint_payload(model, args.checkpoint, source, expected_sha256=args.checkpoint_sha256)
        if payload["counters"]["optimizer_updates"] != 128 or payload["configuration"]["sources"] != training_sources():
            raise ValueError("Require exact current-source fusion startup checkpoint128")
        fixtures, metadata = load_long_fixture(args.fixture, expected_sha256=args.fixture_sha256,
                                              recipe=recipe, width=model.config.model_dim)
        if metadata["training_manifest_sha256"] != payload["configuration"]["data_manifest_sha256"]:
            raise ValueError("Held-out fixture and saved startup data authorities differ")
        optimizer, scheduler = build_optimizer(model)

        def restore():
            return load_fusion_checkpoint(model, args.checkpoint, source, expected_sha256=args.checkpoint_sha256,
                configuration=payload["configuration"], source_fingerprint=payload["source_fingerprint"],
                optimizer=optimizer, scheduler=scheduler)

        report.update(source_checkpoint=source, fixture_provenance=metadata, fixture_pins=fixture_pins(fixtures),
                      original_counters=payload["counters"], original_cursor=payload["data_cursor"],
                      saved_configuration=payload["configuration"])
        restore()

        def publish(row):
            report["rows"].append(row)
            persist(row["case"])
            tracker.log(scalar_metrics(row, "counterfactual_update/"+row["case"]), step=len(report["rows"]))
            print({"case":row["case"], "finite":row["finite"], "elapsed_seconds":time.monotonic()-started}, flush=True)

        result = measure_updates(model, recipe, fixtures, optimizer, scheduler,
            restore=restore, original_flags=original, publish=publish)
        report["comparison"] = {key:value for key,value in result.items() if key != "rows"}
        report["integrity"] = {"sources_unchanged": sources == source_hashes(),
            "checkpoint_unchanged": sha256_file(args.checkpoint) == args.checkpoint_sha256,
            "fixture_unchanged": sha256_file(args.fixture) == args.fixture_sha256,
            "three_optimizer_calls": result["optimizer_calls"] == 3,
            "three_rows": len(report["rows"]) == 3,
            "production_flags_restored": all(getattr(model.backbone.backbone,n) == value for n,value in original.items())}
        if not all(report["integrity"].values()):
            raise AssertionError("Update probe final integrity failed")
        report.update(status="passed_operational_diagnostic", passed=True)
        persist("complete")
    except BaseException as error:
        failure = error
        report.update(status="failed", passed=False, error={"type":type(error).__name__, "message":str(error), "traceback":traceback.format_exc()})
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist(report.get("stage", "setup"))
        try:
            tracker.finish(succeeded=report["passed"])
        except BaseException as error:
            report.update(status="failed", passed=False, tracking_finish_error={"type":type(error).__name__})
            if failure is None:
                raise
        finally:
            persist(report.get("stage", "setup"))


if __name__ == "__main__":
    main()
