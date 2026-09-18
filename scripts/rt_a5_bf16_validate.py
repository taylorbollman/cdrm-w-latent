#!/usr/bin/env python3
"""Bounded protected-BF16 A5 check; all optimizer updates are discarded.

Checks the real restricted-first model and real A5 data. This is not another
precision survey: one small FP32 comparison, three physical-B1024 updates,
T36 inference, and an exact checkpoint/reload continuation check.
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

import torch

from scripts import rt_a5_bf16 as mixed
from scripts import rt_a5_bf16_train as trainer
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import fp32_context, make_optimizer
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_depth_order import build_model as original_build_model
from scripts.rt_a5_nextlat import _parameter_sha256, nextlat_objective
from scripts.rt_a5_train import WordOrder, atomic_json, batch_tensors, cpu_tree, file_sha256, json_value, preserve_rng
from scripts.stage_a_common import require_cuda_container


def require(value, message):
    if not value:
        raise AssertionError(message)


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def finite_state(model, optimizer=None, require_gradients=True):
    failures, names = [], []
    for name, parameter in model.named_parameters():
        names.append(name)
        for role, value in (("parameter", parameter), ("gradient", parameter.grad)):
            if value is None:
                if role == "gradient" and require_gradients:
                    failures.append(f"missing:{name}")
            elif value.dtype != torch.float32 or not torch.isfinite(value).all().item():
                failures.append(f"invalid-{role}:{name}")
    steps = set()
    if optimizer is not None:
        for name, parameter in model.named_parameters():
            state = optimizer.state.get(parameter, {})
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                failures.append(f"missing-adam:{name}")
                continue
            steps.add(int(state["step"].item()))
            for key, value in state.items():
                if value.dtype != torch.float32 or not torch.isfinite(value).all().item():
                    failures.append(f"invalid-adam:{name}:{key}")
    return {"passed": not failures, "failures": failures, "parameter_tensors": len(names),
            "optimizer_steps": sorted(steps)}


def error_metrics(reference, actual):
    ref, got = reference.double(), actual.double()
    norm, maximum = ref.norm().item(), ref.abs().max().item()
    diff = got - ref
    return {"reference_l2": norm, "error_l2": diff.norm().item(),
            "relative_l2": diff.norm().item() / norm if norm else None,
            "max_error_over_reference_max": diff.abs().max().item() / maximum if maximum else None,
            "max_absolute_error": diff.abs().max().item()}


def model_kwargs(config, device="cuda"):
    return {"variant": config["variant"], "width": config["width"], "seed": config["seed"],
            "predictor_seed": config["predictor_seed"], "device": device,
            "predictor_hidden_width": config["predictor_hidden_width"]}


def small_comparison(config, x, y):
    packets, observed = {}, {}
    for arm in ("original_fp32", "adapter_fp32", "protected_bf16"):
        model = (original_build_model if arm == "original_fp32" else mixed.build_model)(**model_kwargs(config))
        if arm == "protected_bf16":
            for layer, block in enumerate(model.backbone.transformer.blocks):
                def observer(phase, tensors, token_index=None, layer=layer):
                    observed.setdefault(str(layer), {}).setdefault(phase, {}).update(
                        {key: str(value.dtype) for key, value in tensors.items() if isinstance(value, torch.Tensor)})
                block._recurrent_precision_observer = observer
        context = mixed.mixed_context if arm == "protected_bf16" else fp32_context
        objective = nextlat_objective if arm == "original_fp32" else mixed.objective
        with context("cuda"):
            result = objective(model, x, y, latent_weight=config["latent_weight"])
        result["loss"].backward()
        state = finite_state(model)
        require(state["passed"], f"Invalid {arm} gradients")
        packets[arm] = {"losses": {key: result[key].detach().cpu() for key in ("loss", "state_loss", "latent_loss")},
                        "logits": result["logits"].detach().cpu(),
                        "gradients": {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()},
                        "finite_state": state, "parameter_sha256": _parameter_sha256(model)}
        del model, result
        gc.collect()
        torch.cuda.empty_cache()
    a, b, c = [packets[key] for key in ("original_fp32", "adapter_fp32", "protected_bf16")]
    exact = {key: tree_equal(a[key], b[key]) for key in ("losses", "logits", "gradients")}
    require(all(exact.values()), "Inactive mixed adapter changed original FP32 computation")
    require(len({packet["parameter_sha256"] for packet in packets.values()}) == 1, "Initialization differs between arms")
    per_tensor = {name: error_metrics(value, c["gradients"][name]) for name, value in a["gradients"].items()}
    global_error = error_metrics(torch.cat([v.flatten() for v in a["gradients"].values()]),
                                 torch.cat([v.flatten() for v in c["gradients"].values()]))
    flagged = {name: value for name, value in per_tensor.items()
               if value["relative_l2"] is None or value["relative_l2"] > 0.03125
               or value["max_error_over_reference_max"] is None or value["max_error_over_reference_max"] > 0.0625}
    for layer in ("0", "1"):
        values = observed[layer]
        require(values["forward.projected"]["q"] == "torch.bfloat16", f"Layer {layer} Q was not BF16")
        require(values["forward.projected"]["x"] == "torch.float32", f"Layer {layer} residual was not FP32")
        require(values["forward.attention"]["attention"] == "torch.float32", f"Layer {layer} attention protection inactive")
    return {"passed": True, "shape": list(x.shape), "fp32_adapter_exact": exact,
            "initial_parameters_exact": True, "finite_state": {k: p["finite_state"] for k, p in packets.items()},
            "losses": {k: {name: value.item() for name, value in p["losses"].items()} for k, p in packets.items()},
            "logits": error_metrics(a["logits"], c["logits"]), "global_gradient": global_error,
            "per_tensor_gradients": per_tensor, "historical_screen_flags": flagged,
            "global_historical_screen_flag": global_error["relative_l2"] is None or global_error["relative_l2"] > 0.015625,
            "screen_scope": "Descriptive historical engineering triggers, not automatic learning acceptance; all coordinates, FP64 reductions",
            "observed_dtypes": observed}


def run(args):
    hardware = require_cuda_container()
    mixed.configure_runtime()
    output, baseline = Path(args.output).resolve(), Path(args.baseline_train).resolve()
    if output.exists():
        raise FileExistsError("Use a new validation output directory")
    config = json.loads((baseline / "config.json").read_text())
    require(config["variant"] == "rt_window2_first" and config["width"] == 512 and config["batch_size"] == 1024,
            "Expected the original two-layer D512/B1024 A5 run")
    data_root = Path(args.data_dir).resolve()
    validate_manifest(data_root)
    train_x, train_y = load_split(data_root, "train")
    ood_x, ood_y = load_split(data_root, "ood_dev")
    require(file_sha256(data_root / "manifest.json") == json.loads((baseline / "report.json").read_text())["contract"]["data_manifest_sha256"],
            "Data differs from historical baseline")
    output.mkdir(parents=True)
    sources = trainer.source_manifest()
    sources["scripts/rt_a5_bf16_validate.py"] = file_sha256(Path(__file__))
    for relative in sources:
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    report = {"schema": "rt-a5-bf16-validation-v1", "status": "running", "hardware": hardware,
              "source_files": sources, "precision_contract": mixed.PRECISION_CONTRACT,
              "scope": "Discarded bounded checks only; no training or confirmation-set selection"}
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name="l1r-nextlat-bf16-preflight", preserve_state=preserve_rng)
    try:
        tracker.start(report)
        report["wandb"] = tracker.record
        model = mixed.build_model(**model_kwargs(config, "cpu"))
        cp0 = baseline / "checkpoints/step-000000.pt"
        packet = torch.load(cp0, map_location="cpu", weights_only=False)
        require(tree_equal(model.state_dict(), packet["model"]), "Original step-0 tensors differ")
        require(json_value(model.nextlat_initialization) == packet["initialization"], "Original initialization metadata differs")
        report["baseline_checkpoint_initialization"] = {
            "path": str(cp0), "sha256": file_sha256(cp0), "state_tensors_exact": True,
            "initialization_exact": True, "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_tensors": len(list(model.parameters())), "model_parameter_sha256": _parameter_sha256(model)}
        del model, packet
        order = WordOrder(len(train_x), config["data_order_seed"])
        x, y = batch_tensors(train_x, train_y, order.indices(0, 2), "cuda")
        report["small_comparison"] = small_comparison(config, x, y)
        atomic_json(output / "report.json", report)
        print(json.dumps({"event": "small_check", "global_gradient": report["small_comparison"]["global_gradient"],
                          "flagged_tensors": list(report["small_comparison"]["historical_screen_flags"])}), flush=True)
        model = mixed.build_model(**model_kwargs(config))
        optimizer = make_optimizer(model)
        report["actual_shape_updates"] = []
        chain = hashlib.sha256(b"rt-a5-word-order-v1").hexdigest()
        for update in (1, 2):
            indices = order.indices((update - 1) * config["batch_size"], config["batch_size"])
            chain = hashlib.sha256(bytes.fromhex(chain) + indices.astype("<i8").tobytes()).hexdigest()
            x, y = batch_tensors(train_x, train_y, indices, "cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            values = mixed.train_step(model, optimizer, x, y, latent_weight=config["latent_weight"], diagnostics=True)
            torch.cuda.synchronize()
            values["seconds"] = time.perf_counter() - started
            finite = finite_state(model, optimizer)
            require(finite["passed"] and finite["optimizer_steps"] == [update], "Invalid physical-shape update")
            report["actual_shape_updates"].append({"update": update, "shape": list(x.shape), "metrics": values, "finite_state": finite})
            tracker.log({"update": update, **{f"preflight/{key}": value for key, value in values.items()}})
        # Strict full-state restoration, then identical changed-input Adam update.
        train_args = argparse.Namespace(**{**config, "stop_file": None})
        contract = trainer.make_contract(train_args, model=model, sources=trainer.source_manifest(),
                                         data_root=data_root, train_x=train_x, hardware=hardware)
        cp = output / "checkpoints/step-000002.pt"
        saved = trainer.save_checkpoint(cp, model=model, optimizer=optimizer, contract=contract,
                                        completed=2, order_chain=chain, initialization=json_value(model.nextlat_initialization))
        restored = mixed.build_model(**model_kwargs(config))
        restored_optimizer = make_optimizer(restored)
        restored_packet = trainer.load_checkpoint(cp, model=restored, optimizer=restored_optimizer, contract=contract)
        require(tree_equal(model.state_dict(), restored.state_dict()) and
                tree_equal(optimizer.state_dict(), restored_optimizer.state_dict()), "Checkpoint restoration differs")
        x, y = batch_tensors(train_x, train_y, order.indices(2 * config["batch_size"], config["batch_size"]), "cuda")
        values = mixed.train_step(model, optimizer, x, y, latent_weight=config["latent_weight"])
        restored_values = mixed.train_step(restored, restored_optimizer, x, y, latent_weight=config["latent_weight"])
        exact = {"model": tree_equal(model.state_dict(), restored.state_dict()),
                 "optimizer": tree_equal(optimizer.state_dict(), restored_optimizer.state_dict()),
                 "metrics": values == restored_values}
        require(all(exact.values()), "Resumed changed-input update differs")
        require(finite_state(restored, restored_optimizer)["passed"], "Invalid resumed state")
        report["resume_check"] = {"passed": True, "checkpoint": saved, "restored_update": restored_packet["completed_updates"],
                                  "next_update": 3, "exact": exact, "order_chain": chain}
        # Predictor must never run in ordinary backbone evaluation.
        def forbidden(*unused):
            raise AssertionError("Backbone-only inference invoked the NextLat predictor")
        handle = model.predictor.register_forward_pre_hook(forbidden)
        try:
            report["length36_evaluation"] = mixed.evaluate_arrays(model, ood_x, ood_y, batch_size=1024, limit=1024)
        finally:
            handle.remove()
        require(all(file_sha256(ROOT / path) == sha for path, sha in sources.items()), "Source changed during preflight")
        report.update(status="passed", peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                      observer_hooks_absent_in_actual_updates=True, confirmation_evaluated=False)
        tracker.summary({"passed": True, "gradient/global_relative_l2": report["small_comparison"]["global_gradient"]["relative_l2"],
                         "gradient/historical_tensor_flags": len(report["small_comparison"]["historical_screen_flags"]),
                         "resume_exact": True})
        tracker.finish(succeeded=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        tracker.finish(succeeded=False)
        raise
    finally:
        report["wandb"] = tracker.record
        atomic_json(output / "report.json", report)
    print(json.dumps({"event": "preflight_complete", "status": report["status"], "output": str(output)}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--baseline-train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wandb-group")
    run(parser.parse_args())
