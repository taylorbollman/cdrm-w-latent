#!/usr/bin/env python3
"""Bounded NUM diagnostics for R3; never advances a research trajectory."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from r3_validation_metrics import compare_tensors
from stage_a_common import (configure_compiled_helpers, provenance,
                            require_cuda_container, seed_all, unique_parameters)
from stage_b_train import aligned_ce_sum, compiler_audit, file_digest


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cpu(value):
    return value.detach().cpu().clone()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def tensors(packet):
    return {**packet["parameters"], **{f"input/{k}": v for k, v in packet["inputs"].items()}}


def compare_maps(reference, actual):
    if set(reference) != set(actual):
        raise AssertionError("Gradient/parameter key sets differ")
    rows = {name: compare_tensors(reference[name], actual[name]) for name in reference}
    return {"tensor_count": len(rows), "elementwise_pass": all(r["elementwise_pass"] for r in rows.values()),
            "scale_aware_pass": all(r["scale_aware_pass"] for r in rows.values()),
            "legacy_failed_tensors": [n for n, r in rows.items() if not r["elementwise_pass"]],
            "scale_failed_tensors": [n for n, r in rows.items() if not r["scale_aware_pass"]],
            "rows": rows}


def forward_graph(model, tokens):
    captured = {}

    def embedding_hook(module, inputs, output):
        output.retain_grad()
        captured["embedding_output"] = output

    def block_hook(module, inputs):
        inputs[0].retain_grad()
        captured["block3_input"] = inputs[0]

    hooks = [model.transformer.wte.register_forward_hook(embedding_hook),
             model.transformer.blocks[3].register_forward_pre_hook(block_hook)]
    try:
        with torch.autocast("cuda", enabled=False):
            logits = model(tokens).logits
        logits.retain_grad()
    finally:
        for hook in hooks:
            hook.remove()
    if set(captured) != {"embedding_output", "block3_input"}:
        raise AssertionError("Missing requested activation capture")
    return logits, captured


def backward_packet(model, graph, *, cotangent=None, labels=None, retain=False):
    logits, captured = graph
    model.zero_grad(set_to_none=True)
    logits.grad = None
    for value in captured.values():
        value.grad = None
    loss_value = None
    if labels is not None:
        summed, count = aligned_ce_sum(logits, labels)
        loss = summed / count
        loss_value = loss.item()
        loss.backward(retain_graph=retain)
    else:
        logits.backward(cotangent, retain_graph=retain)
    params = {n: None if p.grad is None else cpu(p.grad)
              for n, p in model.named_parameters() if p.requires_grad}
    inputs = {n: None if x.grad is None else cpu(x.grad) for n, x in captured.items()}
    return {"parameters": params, "inputs": inputs, "logits": cpu(logits),
            "cotangent": cpu(logits.grad), "loss": loss_value,
            "parameter_order": list(params)}


def conversion(reference, actual):
    from olmo.checkpoint_conversion import convert_model
    mapping = convert_model(reference, actual)
    names_ref = [n for n, p in reference.named_parameters() if p.requires_grad]
    names_actual = [n for n, p in actual.named_parameters() if p.requires_grad]
    expected = {n for item in mapping.parameter_mappings for n in item.source_keys}
    if names_ref != names_actual or set(names_ref) != expected:
        raise AssertionError("Canonical parameter mapping/order is not exhaustive")
    for model in (reference, actual):
        unique_parameters(model)
    if any(not torch.equal(reference.state_dict()[n], actual.state_dict()[n])
           for n in reference.state_dict()):
        raise AssertionError("Starting weights differ")
    return mapping.to_dict()


def ce_cotangent(model, batch):
    tokens = torch.as_tensor(batch.input_ids, device="cuda")
    labels = torch.as_tensor(batch.labels, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        logits = model(tokens).logits
    logits.requires_grad_(True)
    summed, count = aligned_ce_sum(logits, labels)
    cotangent = torch.autograd.grad(summed / count, logits)[0]
    return cotangent.detach(), tokens, labels


def run_raw(args, report):
    from olmo.model import OLMo
    from stage_b_backend_check import config
    from cdrm.synthetic.retrieval import generate_mqar
    prior = json.loads(args.legacy_report.read_text())
    old = prior["arguments"]
    for name, expected in prior["provenance"]["source_sha256"].items():
        if file_digest(Path(name)) != expected:
            raise AssertionError(f"Historical model/support source changed: {name}")
    spec = SimpleNamespace(width=old["width"], length=old["length"], vocab=old["vocab"])
    seed_all(old["seed"], deterministic=True)
    # Preserve historical CUDA initialization and RNG consumption order exactly.
    naive = OLMo(config(spec)).train()
    tiled = OLMo(config(spec, backend="tiled")).train()
    report["conversion"] = conversion(naive, tiled)
    tokens = torch.randint(2, old["vocab"], (old["batch"], old["length"]), device="cuda")
    raw = torch.randn(old["batch"], old["length"], old["vocab"], device="cuda").bfloat16().float()
    for key, value in [("token_sha256", tokens), ("cotangent_sha256", raw)]:
        if tensor_hash(value) != prior["fixture"][key]:
            raise AssertionError(f"Original random fixture not reproduced: {key}")
    state = {n: cpu(v) for n, v in naive.state_dict().items()}
    report["model_config"] = dataclasses.asdict(naive.config)
    if report["model_config"] != prior["model_config"]:
        raise AssertionError("Historical model configuration differs")
    report["historical_model_and_support_source_verified"] = True
    report["legacy_report"] = {"path": str(args.legacy_report), "sha256": file_digest(args.legacy_report),
                               "historical_script_sha256": prior.get("script_sha256")}
    task_config = {"sequence_length": old["length"], "vocab_size": old["vocab"],
                   "num_kv_pairs": min(8, old["length"] // 4), "power_a": .01,
                   "random_non_queries": True, "delay_tokens": 0}
    batch = generate_mqar(task_config, "calibration", 9937, old["batch"])
    ce_v, ce_tokens, ce_labels = ce_cotangent(naive, batch)
    raw_norm = raw.double().norm().item()
    ce_norm = ce_v.double().norm().item()
    alpha_ce = ce_norm / raw_norm
    torch.save({"state_dict": state, "model_config": report["model_config"], "tokens": cpu(tokens),
                "cotangent": cpu(raw), "ce_scale_tokens": cpu(ce_tokens), "ce_scale_labels": cpu(ce_labels),
                "ce_scale_cotangent": cpu(ce_v)}, args.output_dir / "fixture.pt")
    report["fixture"] = {"tokens_sha256": tensor_hash(tokens), "cotangent_sha256": tensor_hash(raw),
                          "raw_cotangent_l2": raw_norm, "ce_cotangent_l2": ce_norm,
                          "ce_scale_alpha": alpha_ce, "ce_scale_task": task_config,
                          "ce_scale_scope": "Masked MQAR mean-CE cotangent norm at the same weights on separate valid task inputs; raw scaling keeps the original forward graph."}
    graphs = {"naive": forward_graph(naive, tokens), "tiled": forward_graph(tiled, tokens)}
    models = {"naive": naive, "tiled": tiled}
    base = {name: backward_packet(model, graphs[name], cotangent=raw, retain=True)
            for name, model in models.items()}
    torch.save(base, args.output_dir / "raw-base-gradients.pt")
    report["logits"] = compare_tensors(base["naive"]["logits"], base["tiled"]["logits"])
    report["base_gradients"] = compare_maps(tensors(base["naive"]), tensors(base["tiled"]))
    report["scaling"] = {}
    for label, alpha in [("1_over_32", 1 / 32), ("32", 32.), ("ce_norm", alpha_ce)]:
        print(f"Scaling {label}: alpha={alpha:.8g}", flush=True)
        packets = {name: backward_packet(model, graphs[name], cotangent=raw * alpha, retain=True)
                   for name, model in models.items()}
        homogeneity = {}
        for name in models:
            normalized = {n: None if v is None else v.double() / alpha
                          for n, v in tensors(packets[name]).items()}
            homogeneity[name] = compare_maps(tensors(base[name]), normalized)
        report["scaling"][label] = {"alpha": alpha,
            "cotangent_l2": (raw * alpha).double().norm().item(),
            "between_backends": compare_maps(tensors(packets["naive"]), tensors(packets["tiled"])),
            "homogeneity_after_dividing_by_alpha": homogeneity}
    del graphs
    if args.fp64_reference:
        from olmo.config import ModelConfig
        cfg = copy.deepcopy(report["model_config"])
        cfg.update(init_device="cpu", recurrent_backend="naive", reference_eager=True)
        oracle = OLMo(ModelConfig(**cfg)).double().cuda().train()
        oracle.load_state_dict({n: v.double() for n, v in state.items()}, strict=True)
        reference = backward_packet(oracle, forward_graph(oracle, tokens), cotangent=raw.double())
        torch.save(reference, args.output_dir / "naive-fp64-gradients.pt")
        report["fp64_reference"] = {name: {
            "logits": compare_tensors(reference["logits"], p["logits"]),
            "gradients": compare_maps(tensors(reference), tensors(p))} for name, p in base.items()}
        report["fp64_scope"] = "Naive autograd only; same FP32-representable weights/cotangent promoted to FP64. Tiled is never labeled FP64."
    report["status"] = "diagnostics_complete"


def adam_state(optimizer, model):
    return {name: {k: cpu(v) if isinstance(v, torch.Tensor) else v
                   for k, v in optimizer.state.get(p, {}).items()}
            for name, p in model.named_parameters()}


def step_packet(model, optimizer, clip):
    before = {n: cpu(p) for n, p in model.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    coefficient = torch.clamp(clip / (norm + 1e-6), max=1.).item()
    clipped = {n: None if p.grad is None else cpu(p.grad) for n, p in model.named_parameters()}
    optimizer.step()
    after = {n: cpu(p) for n, p in model.named_parameters()}
    return {"clip_norm": norm.item(), "clip_coefficient": coefficient, "clipped_gradients": clipped,
            "weights": after, "deltas": {n: after[n].double() - before[n].double() for n in before},
            "state": adam_state(optimizer, model)}


def run_ce(args, report):
    from olmo.config import ModelConfig
    from olmo.model import OLMo
    from cdrm.synthetic.experiment import generate_training_batch
    from stage_b_train import build_model, learning_rate_at_update, source_hashes
    plan = json.loads(args.plan.read_text())
    payload = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (payload["model_config"]["recurrent_layers"] != [3]
                or payload["model_config"]["recurrent_backend"] != "tiled"
                or payload["model_config"]["recurrent_write_rho"] != 1.0
                or payload["identity"]["task"] != "mqar"):
            raise AssertionError("Expected an MQAR R3/rho1/tiled checkpoint")
        for name, expected in payload["identity"]["source_sha256"].items():
            if file_digest(Path(name)) != expected:
                raise AssertionError(f"Frozen training source changed: {name}")
        if payload["identity"]["plan_sha256"] != file_digest(args.plan):
            raise AssertionError("Checkpoint plan mismatch")
        completed = payload["completed_updates"]
        cfg = copy.deepcopy(payload["model_config"])
        cfg["init_device"] = "cpu"
        tiled = OLMo(ModelConfig(**cfg)).cuda().train()
        tiled.load_state_dict(payload["model"], strict=True)
        batch = generate_training_batch(plan, "mqar", completed)
        if batch.sha256 != payload["next_data_sha256"]:
            raise AssertionError("Checkpoint next-data digest differs")
        report["checkpoint"] = {"path": str(args.checkpoint), "sha256": file_digest(args.checkpoint),
                                 "completed_updates": completed, "next_batch_sha256": batch.sha256,
                                 "source_identity_verified": True}
    else:
        plan["model"].update(d_model=32, mlp_hidden_size=128)
        plan["tasks"]["mqar"]["training_conditions"][0].update(sequence_length=32, num_kv_pairs=4)
        completed = 0
        tiled, _ = build_model(plan, "r3", 1024)
        cfg = dataclasses.asdict(tiled.config)
        batch = generate_training_batch(plan, "mqar", 0)
    original_batch_sha = batch.sha256
    if not 1 <= args.batch <= len(batch.input_ids):
        raise ValueError("Requested physical batch must fit the generated batch")
    batch = batch.take(slice(0, args.batch))
    cfg = copy.deepcopy(dataclasses.asdict(tiled.config))
    cfg.update(init_device="cpu", recurrent_backend="naive", reference_eager=True)
    naive = OLMo(ModelConfig(**cfg)).cuda().train()
    report["conversion"] = conversion(tiled, naive)
    settings = plan["training"]
    lr = (learning_rate_at_update(completed, settings) if completed < settings["updates"]
          else payload["optimizer"]["param_groups"][0]["lr"])
    optimizers = {}
    models = {"naive": naive, "tiled": tiled}
    for name, model in models.items():
        opt = torch.optim.AdamW(unique_parameters(model), lr=lr, betas=tuple(settings["betas"]),
            eps=settings["eps"], weight_decay=settings["weight_decay"], foreach=False, fused=False)
        if payload:
            opt.load_state_dict(copy.deepcopy(payload["optimizer"]))
        for group in opt.param_groups:
            group["lr"] = lr
        optimizers[name] = opt
    states = {n: adam_state(o, models[n]) for n, o in optimizers.items()}
    for name in states["naive"]:
        if states["naive"][name].keys() != states["tiled"][name].keys():
            raise AssertionError("Initial optimizer state coverage mismatch")
        for key in states["naive"][name]:
            a, b = states["naive"][name][key], states["tiled"][name][key]
            if not (torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b):
                raise AssertionError("Optimizer initial values differ")
    tokens = torch.as_tensor(batch.input_ids, device="cuda")
    labels = torch.as_tensor(batch.labels, device="cuda")
    packets, steps = {}, {}
    for name, model in models.items():
        print(f"Actual CE {name}: {list(tokens.shape)}, D{model.config.d_model}, checkpoint update {completed}", flush=True)
        packets[name] = backward_packet(model, forward_graph(model, tokens), labels=labels)
        if any(v is None for v in tensors(packets[name]).values()):
            raise AssertionError("Missing intended task gradient")
        steps[name] = step_packet(model, optimizers[name], settings["gradient_clip"])
    torch.save({"tokens": cpu(tokens), "labels": cpu(labels), "gradients": packets, "steps": steps},
               args.output_dir / "ce-and-update-tensors.pt")
    report["model_config"] = dataclasses.asdict(tiled.config)
    report["fixture"] = {"task": "mqar", "shape": list(tokens.shape), "full_batch_sha256": original_batch_sha,
                          "used_batch_sha256": batch.sha256, "answer_count": int((labels != -100).sum()),
                          "label_alignment": "Stage B aligned_ce_sum / total answers, no shift",
                          "global_batch": args.batch, "microbatch": args.batch, "accumulation": False}
    report["optimizer"] = {"name": "AdamW", "lr": lr, "betas": settings["betas"], "eps": settings["eps"],
                           "weight_decay": settings["weight_decay"], "clip": settings["gradient_clip"],
                           "foreach": False, "fused": False, "initial_state_equal": True,
                           "initial_state_steps": sorted({float(s["step"]) for s in states["naive"].values() if "step" in s}),
                           "scope": "One NUM step only; final-checkpoint diagnostic retains last scheduled LR, not a research continuation."}
    report["losses"] = {n: p["loss"] for n, p in packets.items()}
    report["cotangent_l2"] = {n: p["cotangent"].double().norm().item() for n, p in packets.items()}
    report["logits"] = compare_tensors(packets["naive"]["logits"], packets["tiled"]["logits"])
    report["loss_comparison"] = compare_tensors(torch.tensor(packets["naive"]["loss"], dtype=torch.float64),
                                                 torch.tensor(packets["tiled"]["loss"], dtype=torch.float64))
    report["gradients"] = compare_maps(tensors(packets["naive"]), tensors(packets["tiled"]))
    report["clipping"] = {n: {k: s[k] for k in ("clip_norm", "clip_coefficient")} for n, s in steps.items()}
    report["clipped_gradients"] = compare_maps(steps["naive"]["clipped_gradients"], steps["tiled"]["clipped_gradients"])
    report["post_step_weights"] = compare_maps(steps["naive"]["weights"], steps["tiled"]["weights"])
    report["updates"] = compare_maps(steps["naive"]["deltas"], steps["tiled"]["deltas"])
    flagged = []
    for name, row in report["updates"]["rows"].items():
        row["update_screen_pass"] = (row["finite"] and row["rel_l2"] is not None
            and row["max_abs_error"] is not None and row["rel_l2"] <= 1e-3
            and row["max_abs_error"] <= .01 * lr)
        if not row["update_screen_pass"]:
            flagged.append(name)
    report["updates"]["screen_failed_tensors"] = flagged
    moments = {n: {f"{p}/{key}": value for p, state in s["state"].items() for key, value in state.items()}
               for n, s in steps.items()}
    report["optimizer_states"] = compare_maps(moments["naive"], moments["tiled"])
    report["status"] = "diagnostics_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("raw", "ce"), required=True)
    parser.add_argument("--legacy-report", type=Path)
    parser.add_argument("--plan", type=Path, default=Path("configs/stage_b/pilot.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sdpa", choices=("auto", "math"), default="math")
    parser.add_argument("--fp64-reference", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserve prior NUM artifacts; select a new output directory")
    if args.case == "raw" and not args.legacy_report:
        parser.error("raw requires --legacy-report")
    if args.case == "ce" and args.sdpa != "math":
        parser.error("Actual CE must use strict math SDPA")
    args.output_dir.mkdir(parents=True)
    hardware = require_cuda_container()
    seed_all(937, deterministic=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    configure_compiled_helpers(True)
    report = {"schema": "r3-backward-validation-v1", "evidence_class": "NUM", "status": "running",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "provenance": provenance(hardware), "script_sha256": file_digest(Path(__file__)),
              "metrics_script_sha256": file_digest(Path(__file__).with_name("r3_validation_metrics.py")),
              "settings": {"parameters": "FP32", "computation": "FP32", "autocast": False,
                           "tf32": False, "sdpa": args.sdpa, "deterministic": True,
                           "helper_compilation": True, "whole_model_compilation": False,
                           "cuda_graphs": False, "rho": 1, "chunks": 4}}
    started = time.monotonic()
    try:
        from contextlib import nullcontext
        with sdpa_kernel(SDPBackend.MATH) if args.sdpa == "math" else nullcontext():
            (run_raw if args.case == "raw" else run_ce)(args, report)
        report["compiler"] = compiler_audit(True)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except BaseException as error:
        report.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save_json(args.output_dir / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
