"""Small, explicit utilities for NUM/OPS fixtures; not an LM training framework."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "a21b42d2bc292edb86ed1b62cee4bcab809a9d21"
ARTIFACT_WARNING = "NUM/OPS debugging artifact; NOT a research pretrained checkpoint."


def require_cuda_container() -> dict[str, Any]:
    """Fail before initializing a GPU unless the project's execution contract holds."""
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run via scripts/docker_shell.sh; host GPU execution is prohibited.")
    expected = Path("/workspace/cdrm-w-latent")
    if Path.cwd().resolve() != expected or PROJECT_ROOT != expected:
        raise RuntimeError(f"The container working directory must be {expected}.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Stage A scripts support one process on one GPU only.")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the container; no CPU fallback.")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda:0", "used_gpu_count": 1,
        "visible_gpu_count": torch.cuda.device_count(), "name": props.name,
        "total_memory_bytes": props.total_memory,
        "capability": list(torch.cuda.get_device_capability(0)), "nvidia_smi": smi,
    }


def seed_all(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def configure_compiled_helpers(enabled: bool) -> None:
    if enabled:
        # A length-512 tiled scan has more than eight power-of-two helper shapes.
        # Keep a bounded cache and fail rather than silently benchmarking fallback.
        torch._dynamo.config.recompile_limit = 64
        if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
            torch._dynamo.config.fail_on_recompile_limit_hit = True


def validate_compiler_execution(enabled: bool) -> None:
    if not enabled:
        return
    failures = {str(key): count for key, count in torch._dynamo.utils.counters.get("unimplemented", {}).items()
                if count}
    if failures:
        raise RuntimeError(f"Requested compiled helpers encountered unsupported/fallback paths: {failures}")


def git_record(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    try:
        diff = git("diff", "HEAD", "--", ".", ":(exclude)vendors")
        return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}
    except subprocess.CalledProcessError:
        return {"available": False}


def provenance(hardware: dict[str, Any]) -> dict[str, Any]:
    import olmo
    import olmo.model

    sources = [Path(olmo.model.__file__), Path(olmo.__file__).parent / "config.py",
               Path(olmo.__file__).parent / "checkpoint_conversion.py",
               Path(__file__), PROJECT_ROOT / "scripts/benchmark_model.py",
               PROJECT_ROOT / "scripts/stage_a_ops.py"]
    return {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "hardware": hardware, "olmo_import": str(olmo.__file__),
        "upstream_pinned_revision": UPSTREAM_REVISION,
        "project_git": git_record(PROJECT_ROOT),
        "fork_git": git_record(PROJECT_ROOT / "recurrent-transformer"),
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources if path.exists()},
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_dynamo_recompile_limit": torch._dynamo.config.recompile_limit,
        "torch_dynamo_accumulated_recompile_limit": torch._dynamo.config.accumulated_recompile_limit,
        "torch_dynamo_fail_on_recompile_limit_hit": getattr(torch._dynamo.config, "fail_on_recompile_limit_hit", None),
    }


def model_config(preset: str, topology: str, backend: str, rho: float,
                 *, sequence_length: int | None = None, compiled_helpers: bool = False):
    from olmo.config import ModelConfig

    raw = json.loads((PROJECT_ROOT / "configs/stage_a" / f"{preset}.json").read_text())
    raw.update(recurrent_layers=[3] if topology == "r3" else [],
               recurrent_backend=backend, recurrent_write_rho=rho,
               reference_eager=not compiled_helpers)
    if sequence_length is not None:
        raw["max_sequence_length"] = sequence_length
    return ModelConfig(**raw)


def config_dict(config) -> dict[str, Any]:
    return dataclasses.asdict(config)


def experiment_manifest(config, *, evidence: str, budget: dict[str, Any]) -> dict[str, Any]:
    recurrent = config.recurrent_layers or []
    return {
        "evidence_class": evidence, "artifact_warning": ARTIFACT_WARNING,
        "topology_id": "R3" if recurrent else "SEQ", "replacement_indices_zero_based": recurrent,
        "input_state": "ordinary residual input to block 3" if recurrent else "ordinary block inputs",
        "output_state": "processed z_t passed upward" if recurrent else "ordinary block outputs",
        "parameter_ownership": "one canonical owner per parameter; unique optimizer entries",
        "cache_sources": "temporary input K/V, then permanent K/V from (1-rho)*x_t+rho*z_t" if recurrent else "ordinary input K/V",
        "gradient_stops": [], "mask": "causal ALiBi; fixed-length unpadded fixture; no document packing",
        "backend": config.recurrent_backend if recurrent else "sequential",
        "gates": {"rho": config.recurrent_write_rho} if recurrent else {},
        "checkpoint_provenance": "paired random initialization; no pretrained weights",
        "optimizer_policy": "fresh AdamW; no converted optimizer moments",
        "data": "deterministic synthetic token IDs for NUM/OPS only",
        "tokenizer": "none; symbolic IDs (full preset uses T5-shaped vocabulary only)",
        "auxiliary_source": None, "auxiliary_target": None, "budget": budget,
        "resolved_model_config": config_dict(config), "activation_checkpointing": False,
        "gradient_clipping": None,
        "cuda_graphs": False, "model_compilation": False,
        "helper_compilation": not config.reference_eager,
    }


def unique_parameters(model) -> list[torch.nn.Parameter]:
    registrations = [(name, param) for name, param in model.named_parameters(remove_duplicate=False)
                     if param.requires_grad]
    if len(registrations) != len({id(param) for _, param in registrations}):
        raise AssertionError("A trainable parameter has multiple registered owners.")
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) != len({id(p) for p in params}):
        raise AssertionError("Duplicate optimizer parameter identities.")
    # Catch distinct Parameter wrappers over a shared storage as well.
    pointers = [p.untyped_storage().data_ptr() for p in params if p.numel()]
    if len(pointers) != len(set(pointers)):
        raise AssertionError("Distinct optimizer parameters share storage.")
    return params


def adamw(model, learning_rate: float = 1e-3):
    return torch.optim.AdamW(unique_parameters(model), lr=learning_rate,
                             betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                             foreach=False, fused=False)


def target_count(ids: torch.Tensor, target_mask: torch.Tensor | None = None) -> int:
    if ids.ndim != 2:
        raise ValueError("Expected input IDs [batch, sequence].")
    if target_mask is not None:
        if target_mask.shape != ids.shape or target_mask.dtype != torch.bool:
            raise ValueError("target_mask must be boolean and have input_ids shape.")
        return int(target_mask[:, 1:].sum().item())
    return ids.shape[0] * max(ids.shape[1] - 1, 0)


def shifted_ce_sum(logits: torch.Tensor, ids: torch.Tensor,
                   target_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """The mask applies to target positions. Empty supervision is an explicit zero."""
    count = target_count(ids, target_mask)
    if count == 0:
        return logits.sum() * 0.0, 0
    targets = ids[:, 1:].clone()
    if target_mask is not None:
        targets.masked_fill_(~target_mask[:, 1:], -100)
    ce_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    loss = F.cross_entropy(ce_logits[:, :-1].reshape(-1, logits.shape[-1]),
                           targets.reshape(-1), ignore_index=-100, reduction="sum")
    return loss, count


def update(model, optimizer, batches: list[tuple[torch.Tensor, torch.Tensor | None]],
           *, precision: str = "fp32", measure: bool = False) -> dict[str, Any]:
    """Accumulate loss sums divided by the complete update's valid target count."""
    counts = [target_count(ids, mask) for ids, mask in batches]
    total = sum(counts)
    if total == 0:
        optimizer.zero_grad(set_to_none=True)
        return {"skipped_empty_targets": True, "valid_target_tokens": 0,
                "loss": None, "update_seconds": None, "forward_seconds": None}
    events = []
    losses = []
    if measure:
        torch.cuda.synchronize()
        started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for (ids, mask), count in zip(batches, counts):
        if count == 0:
            continue
        if measure:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            logits = model(ids).logits
            if measure:
                end.record()
                events.append((start, end))
            loss_sum, _ = shifted_ce_sum(logits, ids, mask)
        (loss_sum / total).backward()
        losses.append(loss_sum.detach())
        del logits, loss_sum
    optimizer.step()
    if measure:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        forward = sum(start.elapsed_time(end) for start, end in events) / 1000.0
    else:
        elapsed = forward = None
    loss = sum(value.item() for value in losses) / total
    if not np.isfinite(loss):
        raise FloatingPointError(f"Nonfinite training loss: {loss}")
    return {"skipped_empty_targets": False, "valid_target_tokens": total,
            "input_tokens": sum(ids.numel() for ids, _ in batches), "loss": loss,
            "update_seconds": elapsed, "forward_seconds": forward}


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def set_rho(model, rho: float) -> None:
    model.set_recurrent_write_rho(rho)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path: Path, model, optimizer, *, update_index: int,
                    data_offset: int, schedule: dict[str, Any], manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "stage-a-ops-v1", "artifact_warning": ARTIFACT_WARNING,
               "model": model.state_dict(), "model_config": config_dict(model.config),
               "optimizer": optimizer.state_dict(), "rng": rng_state(),
               "update_index": update_index, "data_offset": data_offset,
               "rho": model.config.recurrent_write_rho, "schedule": schedule,
               "manifest": manifest, "optimizer_policy": "exact_resume"}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model, optimizer, *, fresh_optimizer: bool = False) -> dict[str, Any]:
    # This private helper only loads artifacts written by this script in the current run.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["format"] != "stage-a-ops-v1":
        raise ValueError("Not a Stage A smoke checkpoint.")
    expected = config_dict(model.config)
    actual = payload["model_config"].copy()
    actual["recurrent_write_rho"] = expected["recurrent_write_rho"]
    if actual != expected:
        raise ValueError("Exact-resume model configuration mismatch.")
    model.load_state_dict(payload["model"], strict=True)
    set_rho(model, payload["rho"])
    if fresh_optimizer:
        if optimizer.state:
            raise ValueError("The fresh-optimizer branch requires an uninitialized optimizer.")
        payload.update(optimizer_policy="fresh_optimizer_branch", parent_update_index=payload["update_index"],
                       parent_data_offset=payload["data_offset"], update_index=0,
                       branch_processed_sequences=0)
    else:
        optimizer.load_state_dict(payload["optimizer"])
    # Resetting an optimizer does not reset the common parent's data stream.
    # Both continuation arms start at the same stream position and RNG state.
    restore_rng(payload["rng"])
    return payload
