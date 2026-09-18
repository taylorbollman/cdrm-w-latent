#!/usr/bin/env python3
"""Bounded FP32 checks for one mixed-task embedding arm; discard all weights.

This is a small same-function naive/tiled check followed by two actual-shape
updates. It is not a precision survey, convergence run, or throughput benchmark.
Run only after the baseline GPU job and its closeout have finished.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from cdrm.mad_data import FUZZY_TASK, IGNORE_INDEX, load_dataset
from cdrm.rt_nextlat_task_embeddings import build_model, read_configuration, source_manifest as embedding_sources
from cdrm.rt_nextlat_tasks import encode_inputs, task_loss
from scripts import rt_nextlat_a5_fuzzy_train as trainer
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_nextlat import _parameter_sha256
from scripts.rt_a5_train import WordOrder, atomic_json, batch_tensors, file_sha256, json_sha256, json_value, preserve_rng
from scripts.rt_a5_validate import ATOL, RTOL, compare_tensors, finite_state
from scripts.stage_a_common import require_cuda_container


SCHEMA = "rt-nextlat-a5-fuzzy-embeddings-validation-v1"
BATCH_PER_TASK = 2560
ACTUAL_UPDATES = 2
TASK_LENGTHS = {"a5": 12, "fuzzy": 400}
ORDER_SEEDS = {"a5": 5432, "fuzzy": 2026091604}


def source_manifest():
    sources = dict(trainer.source_manifest())
    sources.update(embedding_sources())
    names = (
        "scripts/rt_nextlat_a5_fuzzy_embeddings_validate.py",
        "scripts/rt_a5_validate.py",
    )
    for name in names:
        sources[name] = file_sha256(ROOT / name)
    return dict(sorted(sources.items()))


def _release():
    gc.collect()
    torch.cuda.empty_cache()


def small_backward_check(config, task, length):
    """Compare the same modified architecture, objective and initial tensors."""
    generator = torch.Generator().manual_seed(2026091830 + length)
    vocab = 60 if task == "a5" else 16
    x = encode_inputs(torch.randint(vocab, (2, length), generator=generator).cuda(), task)
    y = torch.randint(vocab, (2, length), generator=generator).cuda()
    fixtures = {"input_ids": x.cpu().numpy(), "local_labels": y.cpu().numpy()}
    packets = {}
    for backend in ("naive", "tiled"):
        model = build_model(config, seed=1234, predictor_seed=1235, fuzzy_seed=1236,
                            device="cuda", backend=backend).train()
        with fp32_context("cuda"):
            result = task_loss(model, x, y, task=task, latent_weight=1.0)
            result["loss"].backward()
        state = finite_state(model, require_gradients=True)
        if not state["passed"]:
            raise AssertionError(f"Invalid {backend} FP32 gradient state for {task}")
        packets[backend] = {
            "initial_parameter_sha256": _parameter_sha256(model),
            "values": {key: result[key].detach().cpu().clone()
                       for key in ("loss", "ce", "latent", "logits")},
            "gradients": {name: p.grad.detach().cpu().clone()
                          for name, p in model.named_parameters()},
            "finite_state": state,
        }
        del result, model
        _release()
    ref, actual = packets["naive"], packets["tiled"]
    if ref["initial_parameter_sha256"] != actual["initial_parameter_sha256"]:
        raise AssertionError("Naive/tiled initialized model tensors differ")
    if ref["gradients"].keys() != actual["gradients"].keys():
        raise AssertionError("Naive/tiled gradient parameter names differ")
    comparisons = {key: compare_tensors(value, actual["values"][key])
                   for key, value in ref["values"].items()}
    gradients = {name: compare_tensors(value, actual["gradients"][name])
                 for name, value in ref["gradients"].items()}
    return {
        "passed": all(value["passed"] for value in (*comparisons.values(), *gradients.values())),
        "task": task, "shape": [2, length], "d_model": 128, "n_heads": 16,
        "initial_parameter_sha256": ref["initial_parameter_sha256"],
        "initial_parameters_exact": True, "values": comparisons, "gradients": gradients,
        "finite_state": {backend: packet["finite_state"] for backend, packet in packets.items()},
        "scope": "Full task-local CE + NextLat, unscaled raw gradients; same modified architecture in both backends",
    }, fixtures


def actual_shape_check(config, arrays, tracker, progress):
    """Exercise the intended physical batch, optimizer and two changed inputs."""
    model = build_model(config, seed=1234, predictor_seed=1235, fuzzy_seed=1236, device="cuda")
    optimizer = make_optimizer(model, lr=1e-4)
    orders = {task: WordOrder(len(arrays[task][0]), ORDER_SEEDS[task]) for task in TASK_LENGTHS}
    report = {"passed": False, "batch_per_task": BATCH_PER_TASK, "microbatch": BATCH_PER_TASK,
              "task_lengths": TASK_LENGTHS, "requested_updates": ACTUAL_UPDATES,
              "completed_updates": 0, "rows": [], "weights_discarded": True,
              "checkpoint_written": False, "initialization": json_value(model.initialization),
              "timing_scope": "Two cold smoke updates only; not a stabilized throughput estimate"}
    _release()
    torch.cuda.reset_peak_memory_stats()
    for index in range(ACTUAL_UPDATES):
        batches, row_hashes = {}, {}
        for task in TASK_LENGTHS:
            selection = orders[task].indices(index * BATCH_PER_TASK, BATCH_PER_TASK)
            row_hashes[task] = hashlib.sha256(selection.astype("<i8").tobytes()).hexdigest()
            x, y = batch_tensors(*arrays[task], selection, "cuda")
            batches[task] = (encode_inputs(x, task), y)
            del x, y
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = trainer.train_step(model, optimizer, batches, mode="mixed",
                                    microbatch=BATCH_PER_TASK, latent_weight=1.0, clip_norm=1.0)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        state = finite_state(model, optimizer, require_gradients=True)
        strict_state = trainer.fuzzy_base.finite_state(model, optimizer, index + 1)
        if not state["passed"] or state["optimizer_steps"] != [index + 1]:
            raise AssertionError("Actual-shape model/gradient/Adam state is invalid")
        row = {"update": index + 1, "seconds": seconds, "loss": result["loss"],
               "grad_norm": result["grad_norm"], "tasks": result["tasks"],
               "ordered_rows_sha256": row_hashes, "finite_state": state,
               "strict_finite_state": strict_state}
        report["rows"].append(row)
        report.update(completed_updates=index + 1,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        tracker.log({"update": index + 1, "train/smoke/loss": result["loss"],
                     "train/smoke/grad_norm": result["grad_norm"],
                     "benchmark/cold_smoke_update_seconds": seconds,
                     **scalar_metrics(result["tasks"], "train/smoke")})
        progress(report)
        del batches, result
    report.update(passed=True, peak_allocated_gib=report["peak_allocated_bytes"] / 2**30,
                  peak_reserved_gib=report["peak_reserved_bytes"] / 2**30)
    del model, optimizer
    _release()
    return report


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Validation requires a fresh output directory")
    config = read_configuration(args.config)
    raw = config["backbone"]
    if (raw["d_model"], raw["n_heads"], raw["mlp_hidden_size"], raw["n_layers"]) != (128, 16, 512, 2):
        raise ValueError("This check requires the two-layer D128/H16/FFN512 mixed pilot")
    injection = config.get("embedding_injection")
    if not injection or injection["variant"] not in ("input", "value", "head"):
        raise ValueError("Select one embedding variant, not the historical baseline")
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    a5_root, fuzzy_root = Path(args.a5_data).resolve(), Path(args.fuzzy_data).resolve()
    a5_manifest = validate_manifest(a5_root)
    if file_sha256(a5_root / "manifest.json") != trainer.A5_MANIFEST_SHA256:
        raise ValueError("Expected the frozen A5 corpus")
    a5 = load_split(a5_root, "train")
    fuzzy = load_dataset(fuzzy_root, FUZZY_TASK, "train")
    arrays = {"a5": a5, "fuzzy": (fuzzy.input_ids, fuzzy.labels)}
    if any(arrays[task][0].shape[1] != length for task, length in TASK_LENGTHS.items()):
        raise ValueError("Expected native A5 T12 and Fuzzy T400 shapes")
    if np.any(fuzzy.labels == IGNORE_INDEX):
        raise ValueError("Fuzzy smoke training retains the baseline's native dense targets")
    sources = source_manifest()
    output.mkdir(parents=True)
    for relative, expected in sources.items():
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if file_sha256(target) != expected:
            raise RuntimeError("Validation source changed during snapshot")
    shutil.copy2(args.config, output / "model-config.json")
    atomic_json(output / "source-manifest.json", sources)
    atomic_json(output / "invocation.json", vars(args))
    contract = {"schema": SCHEMA, "model_config": config,
                "configuration_file_sha256": file_sha256(args.config),
                "source_sha256": json_sha256(sources), "hardware": hardware, "runtime": runtime,
                "atol": ATOL, "rtol": RTOL, "batch_per_task": BATCH_PER_TASK,
                "microbatch": BATCH_PER_TASK, "actual_updates": ACTUAL_UPDATES,
                "order_seeds": ORDER_SEEDS,
                "data": {"a5_manifest": a5_manifest, "a5_manifest_sha256": trainer.A5_MANIFEST_SHA256,
                         "fuzzy_train": fuzzy.manifest,
                         "fuzzy_preparation_manifest_sha256": file_sha256(fuzzy_root / "manifest.json")},
                "scope": "Small same-function naive/tiled FP32 task losses/gradients and two full-B2560 mixed optimizer updates; all weights discarded",
                "not_tested": ["Full-B2560 naive reference", "Trained-state numerical parity",
                               "Mixed precision or compilation", "Learning or stable throughput"]}
    report = {"schema": SCHEMA, "status": "running", "contract": contract,
              "small_backward": {}, "checkpoint_written": False, "confirmation_evaluated": False}
    tracker = OnlineTracker(project=args.wandb_project, entity="taylorbollman", output_dir=output,
                            group=args.wandb_group,
                            name=args.wandb_run_name or f"d128-{injection['variant']}-embedding-preflight",
                            preserve_state=preserve_rng)
    report["wandb"] = tracker.record
    atomic_json(output / "report.json", report)
    started, succeeded = time.perf_counter(), False
    try:
        tracker.start(contract)
        for task, length in (("a5", 12), ("fuzzy", 17)):
            check, fixture = small_backward_check(config, task, length)
            path = output / f"{task}-small-fixture.npz"
            np.savez(path, **fixture)
            check["fixture"] = {"path": path.name, "sha256": file_sha256(path)}
            report["small_backward"][task] = check
            atomic_json(output / "report.json", report)
            if not check["passed"]:
                raise AssertionError(f"Naive/tiled {task} gradient/logit comparison failed")

        def progress(record):
            report.update(actual_shapes=record, elapsed_seconds=time.perf_counter() - started)
            atomic_json(output / "report.json", report)

        report["actual_shapes"] = actual_shape_check(config, arrays, tracker, progress)
        if source_manifest() != sources or file_sha256(args.config) != contract["configuration_file_sha256"]:
            raise RuntimeError("Validation source/config changed during execution")
        report.update(status="passed", elapsed_seconds=time.perf_counter() - started)
        tracker.summary({"passed": True, "variant": injection["variant"],
                         "peak_allocated_gib": report["actual_shapes"]["peak_allocated_gib"],
                         "peak_reserved_gib": report["actual_shapes"]["peak_reserved_gib"]})
        succeeded = True
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__,
                      elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        try:
            tracker.finish(succeeded=succeeded)
        finally:
            report["wandb"] = tracker.record
            atomic_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "variant": injection["variant"],
                      "wandb": tracker.record.get("run_url"),
                      "peak_allocated_gib": report["actual_shapes"]["peak_allocated_gib"]}), flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--a5-data", type=Path, required=True)
    result.add_argument("--fuzzy-data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--wandb-project", default="rt-nextlat-fuzzy-a5")
    result.add_argument("--wandb-group")
    result.add_argument("--wandb-run-name")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
