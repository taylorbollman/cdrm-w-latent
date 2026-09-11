"""Shared state, data and execution contract for five-block tiled CDRM validation."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.mad_data import load_dataset, epoch_indices
from cdrm_common import optimizer_for, state_summaries, save_checkpoint
from r3_mixed_operational import cpu_tree, state_digest, effective_precision, execution_contract
from stage_a_common import (require_cuda_container, seed_all, unique_parameters,
                            configure_compiled_helpers, provenance, rng_state, restore_rng)
from stage_b_train import atomic_json, append_jsonl, file_digest, json_digest, compiler_audit

PRESET = Path("configs/cdrm/base_d128_5.json")
TASK = "selective-copying"
INIT_FORMAT = "cdrm-tiled-initialization-v1"
FORMAT = "cdrm-tiled-operational-v1"
POLICIES = {"fp32": "fp32", "bf16": "bf16_fp32_state", "bf16_fp32_state": "bf16_fp32_state"}
ARM_FIELDS = {"cdrm_backend", "cdrm_precision_policy", "reference_eager"}


@contextmanager
def preserve_tracking_rng():
    saved = rng_state()
    try:
        yield
    finally:
        restore_rng(saved)


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def setup(seed=0):
    hardware = require_cuda_container()
    torch.set_num_threads(1)
    seed_all(seed, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_autocast_cache_enabled(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    configure_compiled_helpers(True)
    runtime = execution_contract()
    runtime.update(autocast_cache=True, cudnn_sdpa=False,
                   supported_mixed_policy="BF16 dense; FP32 residuals, norms, differences, recurrent state and optimizer",
                   torch_num_threads=torch.get_num_threads())
    return {"provenance": provenance(hardware), "execution_contract": runtime}


def sources(extra_paths=()):
    paths = set(Path("recurrent-transformer/olmo").rglob("*.py"))
    paths.update(Path(name) for name in (
        "scripts/cdrm_tiled_common.py", "scripts/cdrm_tiled_operational.py",
        "scripts/cdrm_common.py", "scripts/cdrm_train.py", "scripts/experiment_tracking.py",
        "scripts/stage_a_common.py", "scripts/stage_b_train.py",
        "scripts/r3_mixed_operational.py", "scripts/r3_validation_metrics.py",
        "cdrm/mad_data.py", "vendors/mad-lab/mad/data/instances.py", str(PRESET)))
    paths.update(Path(name) for name in extra_paths)
    return {str(path): file_digest(path) for path in sorted(paths) if path.is_file()}


def snapshot_sources(directory, tracked):
    for name, expected in tracked.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Source snapshot paths must be relative to the project")
        destination = Path(directory) / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(path.read_bytes())
        if file_digest(destination) != expected:
            raise RuntimeError(f"Source changed while snapshotting: {name}")


def verify_sources(tracked):
    changed = [name for name, expected in tracked.items()
               if not Path(name).is_file() or file_digest(Path(name)) != expected]
    if changed:
        raise RuntimeError(f"Experiment source identity changed: {changed}")


def config(backend="tiled", precision="fp32", *, preset=PRESET, eager=False, overrides=None):
    from olmo.config import ModelConfig
    if backend not in ("naive", "tiled") or precision not in POLICIES:
        raise ValueError("Require naive/tiled backend and an explicit supported precision")
    policy = POLICIES[precision]
    if backend == "naive" and policy != "fp32":
        raise ValueError("The naive CDRM oracle remains strictly FP32")
    raw = json.loads(Path(preset).read_text())
    raw.update(cdrm_enabled=True, cdrm_backend=backend, cdrm_precision_policy=policy,
               cdrm_source="deep", cdrm_read_mode="history", cdrm_epsilon=.1,
               cdrm_rho=1., cdrm_lambda=.01, cdrm_output_states=False,
               recurrent_layers=[], recurrent_backend="naive", recurrent_precision_policy="legacy",
               reference_eager=backend == "naive" or eager, init_device="cpu", precision=None)
    raw.update(overrides or {})
    return ModelConfig(**raw)


def cpu_initial_model(seed=7500, *, preset=PRESET):
    """Match the prior common-backbone/adapters seed isolation without CUDA work."""
    from olmo.model import OLMo
    seed_cpu(seed)
    base_config = config("naive", "fp32", preset=preset)
    base_config.cdrm_enabled = False
    base = OLMo(base_config)
    common = cpu_tree(base.state_dict())
    backbone_digest = state_digest(common)
    seed_cpu(seed + 100003)
    model = OLMo(config("naive", "fp32", preset=preset))
    target = model.state_dict()
    extras = sorted(set(target) - set(common))
    if extras != ["cdrm.bridge_adapter.weight", "cdrm.deep_adapter.weight"]:
        raise AssertionError(f"Unexpected additional parameter owners: {extras}")
    for name, value in common.items():
        if target[name].shape != value.shape:
            raise AssertionError(f"Backbone shape differs: {name}")
        target[name] = value
    model.load_state_dict(target, strict=True)
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in common.items()):
        raise AssertionError("Backbone initialization copy was not exact")
    unique_parameters(model)
    return model, {"seed": seed, "adapter_seed": seed + 100003, "training_seed": seed + 200003,
                   "backbone_initialization_sha256": backbone_digest,
                   "full_initialization_sha256": state_digest(model.state_dict()),
                   "additional_parameter_owners": extras}


def build_model(backend="tiled", precision="fp32", seed=7500, *, checkpoint=None,
                eager=False, device="cuda", preset=PRESET):
    """Load identical full weights; permit only the three declared arm controls."""
    from olmo.config import ModelConfig
    from olmo.model import OLMo
    if backend not in ("naive", "tiled") or precision not in POLICIES:
        raise ValueError("Unsupported CDRM execution arm")
    if backend == "naive" and POLICIES[precision] != "fp32":
        raise ValueError("Naive CDRM is an FP32 reference only")
    if checkpoint is None:
        initial, construction = cpu_initial_model(seed, preset=preset)
        payload = {"model_config": dataclasses.asdict(initial.config), "model": cpu_tree(initial.state_dict())}
        del initial
    else:
        payload = (torch.load(checkpoint, map_location="cpu", weights_only=False)
                   if isinstance(checkpoint, (str, Path)) else checkpoint)
        construction = copy.deepcopy(payload.get("initialization", {}))
        if "identity" in payload and "source_sha256" in payload["identity"]:
            verify_sources(payload["identity"]["source_sha256"])
        if payload.get("format") not in (None, INIT_FORMAT, FORMAT):
            raise ValueError("Only this five-block initialization/operational checkpoint lineage is supported")
    raw = copy.deepcopy(payload["model_config"])
    raw.update(cdrm_backend=backend, cdrm_precision_policy=POLICIES[precision],
               reference_eager=backend == "naive" or eager)
    if any(raw.get(name) != value for name, value in payload["model_config"].items() if name not in ARM_FIELDS):
        raise AssertionError("Undeclared cross-arm configuration change")
    model = OLMo(ModelConfig(**raw)).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model"], strict=True)
    if state_digest(model.state_dict()) != state_digest(payload["model"]):
        raise AssertionError("Weights changed while building a comparison arm")
    params = unique_parameters(model)
    if len(params) != len(list(model.parameters())):
        raise AssertionError("Duplicate optimizer owners")
    if str(device).startswith("cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    construction.update(model_config=dataclasses.asdict(model.config),
                        parameter_count=sum(p.numel() for p in params),
                        parameter_bytes=sum(p.numel() * p.element_size() for p in params),
                        starting_weights_sha256=state_digest(payload["model"]),
                        allowed_arm_changes=sorted(ARM_FIELDS))
    return model, construction


def dataset_identity(dataset):
    return {"array_sha256": dataset.sha256, "examples": len(dataset),
            "length": dataset.input_ids.shape[1], "manifest_sha256": dataset.manifest.get("manifest_sha256"),
            "vocab_size": dataset.manifest["vocab_size"],
            "native_supervision": dataset.manifest["native_training_objective"]}


def validate_data(dataset):
    if (dataset.manifest["task"] != TASK or dataset.input_ids.shape[1] != 256
            or dataset.manifest["vocab_size"] != 16
            or dataset.manifest.get("task_overrides", {}).get("num_tokens_to_copy") != 96
            or not np.array_equal(dataset.labels, dataset.answer_labels)
            or not np.all((dataset.labels != -100).sum(-1) == 96)):
        raise ValueError("Require original aligned MAD selective-copying V16/T256/K96 supervision")


def loss_sum(logits, labels):
    if logits.dtype not in (torch.float32, torch.bfloat16):
        raise AssertionError("Expected FP32 or BF16 logits with explicit FP32 CE")
    valid = labels != -100
    count = int(valid.sum().item())
    if count == 0 or not valid.any(-1).all().item():
        raise ValueError("Every example requires scored targets")
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction="sum"), count


def metric_counts(logits, labels):
    summed, count = loss_sum(logits, labels)
    valid = labels != -100
    correct = (logits.argmax(-1) == labels) & valid
    return {"ce_sum": float(summed.detach().item()), "targets": count,
            "correct": int(correct.sum().item()), "exact": int((correct | ~valid).all(-1).sum().item()),
            "examples": labels.shape[0]}


def finish_metrics(counts):
    return {**counts, "ce": counts["ce_sum"] / counts["targets"],
            "token_accuracy": counts["correct"] / counts["targets"],
            "sequence_exact_match": counts["exact"] / counts["examples"]}


def update(model, opt, ids, labels, answer_labels=None, *, precision="fp32", monitor=True, clip=1.):
    mixed = POLICIES[precision] == "bf16_fp32_state"
    model.train()
    before = [p.detach().clone() for p in model.parameters()] if monitor else None
    torch.cuda.synchronize()
    started = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
        output = model(ids, output_cdrm_states=monitor)
    logits = output.logits
    expected = torch.bfloat16 if mixed else torch.float32
    if logits.dtype != expected:
        raise AssertionError(f"Expected observed {expected} logits")
    summed, count = loss_sum(logits, labels)
    loss = summed / count
    loss.backward()
    params = list(model.named_parameters())
    if any(p.dtype != torch.float32 or p.grad is None or p.grad.dtype != torch.float32 for _, p in params):
        raise AssertionError("Every intended owner must receive an FP32 gradient")
    signals = {}
    if monitor:
        for name, parameter in params:
            if name.startswith("cdrm.") or name.startswith(f"transformer.blocks.{model.config.cdrm_early_layer}."):
                signals[name] = {"l2": float(parameter.grad.detach().double().norm().item()),
                                 "nonzero": bool(torch.count_nonzero(parameter.grad).item())}
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True).item())
    opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    row = {"native_loss": float(loss.item()), "native_targets": count, "seconds": elapsed,
           "gradient_norm": gradient_norm, "clipped": gradient_norm > clip,
           "clip_coefficient": min(1., clip / (gradient_norm + 1e-6)), "input_tokens": ids.numel(),
           "learning_rate": opt.param_groups[0]["lr"], "logits_dtype": str(logits.dtype), "ce_dtype": str(loss.dtype)}
    if not math.isfinite(row["native_loss"]) or loss.dtype != torch.float32:
        raise FloatingPointError("Native masked CE must be finite FP32")
    if monitor:
        row.update(precision=effective_precision(model, opt, require_gradients=True),
                   state_summaries=state_summaries(output.cdrm_states), parameter_gradient_signals=signals,
                   update_l2=float(torch.stack([(p.detach().double() - old.double()).square().sum()
                                                for p, old in zip(model.parameters(), before)]).sum().sqrt().item()),
                   answer=finish_metrics(metric_counts(logits.detach(), answer_labels if answer_labels is not None else labels)))
    return row


@torch.no_grad()
def evaluate(model, dataset, batch_size=64, *, precision="fp32"):
    model.eval()
    totals = {kind: {"ce_sum": 0., "targets": 0, "correct": 0, "exact": 0, "examples": 0}
              for kind in ("native", "answer")}
    started = time.perf_counter()
    for begin in range(0, len(dataset), batch_size):
        ids = torch.as_tensor(dataset.input_ids[begin:begin + batch_size], device="cuda")
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=POLICIES[precision] == "bf16_fp32_state"):
            logits = model(ids).logits
        for name, array in (("native", dataset.labels), ("answer", dataset.answer_labels)):
            labels = torch.as_tensor(array[begin:begin + batch_size], device="cuda")
            for key, value in metric_counts(logits, labels).items():
                totals[name][key] += value
    torch.cuda.synchronize()
    return {**{name: finish_metrics(values) for name, values in totals.items()},
            "seconds": time.perf_counter() - started}
