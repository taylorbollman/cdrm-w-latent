"""Matched two-block A5 models, strict FP32 execution, loss and metrics.

This task predicts the state *after* the operation at the same position.
The model's native head is used without a language-model target shift.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterator

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from olmo.checkpoint_conversion import convert_model
from olmo.config import ModelConfig
from olmo.model import OLMo


ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 60
MAX_LENGTH = 36


def parameter_count(width: int) -> int:
    """Two bias-free GELU blocks, four learned norms/block, head and embedding."""
    return 24 * width * width + 129 * width


def model_config(architecture: str, width: int = 512, *, backend: str = "tiled",
                 device: str = "cpu") -> ModelConfig:
    if architecture not in ("seq", "rt"):
        raise ValueError("architecture must be 'seq' or 'rt'")
    if type(width) is not int or width < 64 or width % 64:
        raise ValueError("width must be a positive multiple of 64 (head dimension 64)")
    if backend == "autograd":
        backend = "naive"
    if backend not in ("tiled", "naive"):
        raise ValueError("backend must be 'tiled' or 'naive'")
    raw = json.loads((ROOT / "configs/rt_a5/base.json").read_text())
    raw.update(d_model=width, n_heads=width // 64, n_kv_heads=width // 64,
               mlp_hidden_size=4 * width, init_device=str(device), recurrent_backend=backend,
               block_type="sequential" if architecture == "seq" else
               ("recurrent" if backend == "tiled" else "recurrent_autograd"))
    return ModelConfig(**raw)


def canonical_parameter_sha256(model: OLMo) -> str:
    """Hash parameter values in ordinary fused-QKV layout, including all tensors."""
    values = {name: value.detach().cpu() for name, value in model.named_parameters()}
    for index, block in enumerate(model.transformer.blocks):
        if not hasattr(block, "q_proj"):
            continue
        prefix = f"transformer.blocks.{index}."
        for suffix in ("weight", "bias"):
            query, key_value = prefix + "q_proj." + suffix, prefix + "kv_proj." + suffix
            if query in values:
                values[prefix + "att_proj." + suffix] = torch.cat(
                    (values.pop(query), values.pop(key_value)), dim=0)
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_model(architecture: str, width: int = 512, seed: int = 0,
                device: str | torch.device = "cpu", *, backend: str = "tiled") -> OLMo:
    """Initialize a canonical CPU SEQ, then exhaustively convert the RT weights.

    Initialization is independent of the training RNG and accelerator. The
    attached plain dictionary ``a5_initialization`` is provenance, not state.
    """
    target_config = model_config(architecture, width, backend=backend)
    # fork_rng with no accelerator devices preserves the caller's CPU stream;
    # seed the CPU generator directly, without changing CUDA RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        source = OLMo(model_config("seq", width)).to(dtype=torch.float32)
        initial_hash = canonical_parameter_sha256(source)
        if architecture == "seq":
            model, conversion = source, None
        else:
            model = OLMo(target_config).to(dtype=torch.float32)
            conversion = convert_model(source, model).to_dict()
        if canonical_parameter_sha256(model) != initial_hash:
            raise AssertionError("Mapped initialization differs from the canonical SEQ")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != parameter_count(width):
        raise AssertionError(f"Expected {parameter_count(width)} parameters, constructed {count}")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise AssertionError("A5 model parameters must be FP32")
    model.a5_initialization = {
        "schema": "rt-a5-initialization-v1", "seed": seed, "width": width,
        "architecture": architecture, "parameter_count": count,
        "canonical_sha256": initial_hash, "conversion": conversion,
        "rule": "CPU Mitchell SEQ initialization; checked weights-only conversion for RT",
    }
    return model.to(device=device, dtype=torch.float32)


def configure_fp32_runtime() -> dict:
    """Call before model work; the per-forward context also selects math SDPA."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return {
        "precision": "fp32", "autocast": False, "tf32": False,
        "compile": False, "reference_eager": True, "cuda_graphs": False,
        "ordinary_attention_backend": "math", "deterministic_algorithms": True,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
    }


@contextmanager
def fp32_context(device: str | torch.device = "cuda") -> Iterator[None]:
    """Disable any enclosing autocast and select FP32 ordinary math attention.

    OLMo construction changes global SDPA flags, so select this after building
    models and around each forward/backward, not just once before construction.
    """
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        yield


def _validate_shapes(logits: torch.Tensor, labels: torch.Tensor) -> None:
    if logits.ndim != 3 or logits.shape[-1] != VOCAB_SIZE or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch, length, 60] and labels [batch, length]")
    if not labels.numel():
        raise ValueError("A5 batches must be nonempty")
    if logits.dtype != torch.float32 or labels.dtype != torch.long:
        raise TypeError("A5 requires FP32 logits and int64 labels")


def task_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Unshifted CE, averaged over every operation in every word."""
    _validate_shapes(logits, labels)
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), reduction="mean")


class A5Metrics:
    """Exact integer accuracy totals and FP64 CE sums, independent of batching."""

    def __init__(self) -> None:
        self.rows = 0
        self.length: int | None = None
        self.correct: torch.Tensor | None = None
        self.prefix_correct: torch.Tensor | None = None
        self.ce_sum: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        _validate_shapes(logits, labels)
        batch, length = labels.shape
        if self.length is None:
            self.length = length
            self.correct = torch.zeros(length, dtype=torch.int64)
            self.prefix_correct = torch.zeros(length, dtype=torch.int64)
            self.ce_sum = torch.zeros(length, dtype=torch.float64)
        elif self.length != length:
            raise ValueError("Use a separate A5Metrics accumulator for each word length")
        correct = logits.argmax(dim=-1).eq(labels).to(torch.int64)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1),
                             reduction="none").reshape(batch, length)
        self.correct += correct.sum(dim=0).cpu()
        self.prefix_correct += correct.cumprod(dim=1).sum(dim=0).cpu()
        self.ce_sum += ce.double().sum(dim=0).cpu()
        self.rows += batch

    def compute(self) -> dict:
        if not self.rows:
            raise ValueError("Cannot compute A5 metrics without examples")
        accuracy = self.correct.double() / self.rows
        exactness = self.prefix_correct.double() / self.rows
        ce = self.ce_sum / self.rows
        return {
            "rows": self.rows, "length": self.length, "tokens": self.rows * self.length,
            "ce": float(ce.mean()), "token_accuracy": float(accuracy.mean()),
            "final_state_accuracy": float(accuracy[-1]),
            "whole_word_exact_match": float(exactness[-1]),
            "isolated_state_accuracy": accuracy.tolist(),
            "cumulative_prefix_exactness": exactness.tolist(), "per_position_ce": ce.tolist(),
        }


def make_optimizer(model: OLMo, lr: float = 1e-4) -> torch.optim.AdamW:
    """Paper AdamW recipe: decay matrices, excluding all 1-D norm parameters."""
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    matrices, vectors = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (matrices if parameter.ndim >= 2 else vectors).append(parameter)
    return torch.optim.AdamW(
        [{"params": matrices, "weight_decay": 0.01},
         {"params": vectors, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8, foreach=False, fused=False,
    )
