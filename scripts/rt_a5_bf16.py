"""Protected BF16 execution for the unchanged two-layer restricted-first A5 RT.

Parameters, residuals and Adam state stay FP32. Dense operations use CUDA BF16
autocast; the existing bf16_fp32_state kernel protects recurrent attention.
The historical FP32 factory, objective, and training sources are unmodified.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.rt_a5_common import A5Metrics, configure_fp32_runtime, task_loss
from scripts.rt_a5_depth_order import build_model as original_build_model
from scripts.rt_a5_nextlat import _one_step_diagnostics, _parameter_sha256
from scripts.rt_a5_nextlat_train import objective_metrics
from scripts.rt_a5_train import batch_tensors


POLICY = "bf16_fp32_state"
PRECISION_CONTRACT = {
    "precision": "bf16_mixed", "autocast_dtype": "bfloat16",
    "autocast_cache_enabled": True, "recurrent_precision_policy": POLICY,
    "parameters": "float32", "parameter_gradients": "float32",
    "adam_moments": "float32", "residual_stream": "float32",
    "head_logits": "BF16 dense result converted to FP32 before CE and metrics",
    "latent_loss": "FP32 SmoothL1 reduction; BF16 predictor dense operations",
    "loss_scaler": False, "evaluation_precision": "same protected BF16 as training",
    "tf32": False, "compile": False, "cuda_graphs": False,
    "reference_eager": True, "ordinary_attention_backend": "math",
    "backward_autocast": "outside outer autocast; tiled backward restores forward precision",
}


def configure_runtime():
    configure_fp32_runtime()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    return copy.deepcopy(PRECISION_CONTRACT)


@contextmanager
def mixed_context(device="cuda"):
    if torch.device(device).type != "cuda":
        raise ValueError("BF16 A5 execution requires CUDA; no CPU fallback")
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True), sdpa_kernel(SDPBackend.MATH):
        yield


def build_model(variant="rt_window2_first", **kwargs):
    if variant != "rt_window2_first":
        raise ValueError("This precision repeat is restricted to the original L1R RT")
    model = original_build_model(variant, **kwargs)
    before = _parameter_sha256(model)
    # The windowed block has a deep-copied config. Configure every block explicitly.
    for config in [model.backbone.config, *(b.config for b in model.backbone.transformer.blocks)]:
        config.recurrent_precision_policy = POLICY
        if config.precision != "fp32" or not config.reference_eager:
            raise AssertionError("Expected FP32 parameters and unchanged eager execution")
    if _parameter_sha256(model) != before:
        raise AssertionError("Precision configuration changed initialized parameters")
    model.precision_contract = copy.deepcopy(PRECISION_CONTRACT)
    # Preserve original initialization metadata verbatim: it describes the draw,
    # not execution. Runtime precision is recorded separately in the run contract.
    return model


def objective(model, inputs, labels, latent_weight=1.0, diagnostics=False):
    """Original CE + one-step NextLat math, with explicit FP32 loss operands."""
    if not isinstance(latent_weight, (int, float)) or not math.isfinite(latent_weight) or latent_weight < 0:
        raise ValueError("latent_weight must be finite and nonnegative")
    if inputs.ndim != 2 or inputs.shape != labels.shape or not inputs.numel():
        raise ValueError("Inputs and same-position labels must be nonempty [batch, length] tensors")
    if inputs.dtype != torch.long or labels.dtype != torch.long:
        raise TypeError("A5 inputs and labels must have int64 dtype")
    if latent_weight > 0 and inputs.shape[1] < 2:
        raise ValueError("A positive NextLat objective needs at least two operations")
    output = model.backbone(inputs, return_pre_logits=latent_weight > 0)
    logits = output.logits.float()
    state_loss = task_loss(logits, labels)
    if latent_weight == 0:
        return {"loss": state_loss, "state_loss": state_loss,
                "latent_loss": state_loss.new_zeros(()), "logits": logits, "diagnostics": {}}
    hidden = output.pre_logits
    if hidden is None or hidden.dtype != torch.float32:
        raise TypeError("NextLat requires attached FP32 post-final-normalization latents")
    next_embeddings = model.backbone.transformer.wte(inputs[:, 1:])
    predicted = model.predictor(hidden[:, :-1], next_embeddings)
    with torch.autocast(inputs.device.type, enabled=False):
        latent_loss = F.smooth_l1_loss(predicted.float(), hidden[:, 1:].detach(), beta=1.0, reduction="mean")
        diagnostic_values = (_one_step_diagnostics(model, hidden, predicted.float(), logits, labels)
                             if diagnostics else {})
    return {"loss": state_loss + latent_weight * latent_loss,
            "state_loss": state_loss, "latent_loss": latent_loss,
            "logits": logits, "diagnostics": diagnostic_values}


def train_step(model, optimizer, inputs, labels, *, latent_weight=1.0,
               clip_norm=1.0, diagnostics=False):
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with mixed_context(inputs.device):
        result = objective(model, inputs, labels, latent_weight, diagnostics)
        values = objective_metrics(result, latent_weight)
    result["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        correct = result["logits"].argmax(-1).eq(labels)
        values.update(token_accuracy=correct.float().mean().item(),
                      whole_word_exact=correct.all(dim=1).float().mean().item(),
                      grad_norm=grad_norm.item())
    values.update(values.pop("diagnostics"))
    return values


def evaluate_arrays(model, inputs, labels, *, batch_size=1024, limit=None, device="cuda"):
    rows = len(inputs) if limit is None else min(len(inputs), limit)
    if rows < 1 or batch_size < 1:
        raise ValueError("Evaluation needs nonempty data and positive batch size")
    metrics, training, started = A5Metrics(), model.training, time.perf_counter()
    model.eval()
    try:
        with torch.no_grad():
            for begin in range(0, rows, batch_size):
                x, y = batch_tensors(inputs, labels, slice(begin, min(begin + batch_size, rows)), device)
                with mixed_context(device):
                    logits = model(x).logits.float()
                # Same historical metrics implementation, including FP64 CE sums.
                metrics.update(logits, y)
        result = metrics.compute()
    finally:
        model.train(training)
    result.update(evaluation_seconds=time.perf_counter() - started, evaluated_rows=rows)
    return result


def evaluate_diagnostics(model, inputs, labels, *, rows=1024, device="cuda", latent_weight=1.0):
    rows = min(rows, len(inputs))
    if rows < 1:
        raise ValueError("Diagnostic evaluation needs positive rows")
    training = model.training
    model.eval()
    try:
        with torch.no_grad(), mixed_context(device):
            x, y = batch_tensors(inputs, labels, slice(0, rows), device)
            result = objective(model, x, y, latent_weight=latent_weight, diagnostics=True)
            values = objective_metrics(result, latent_weight)
        return {"rows": rows, "length": int(x.shape[1]),
                "route": "teacher_conditioned_one_step_diagnostics", **values}
    finally:
        model.train(training)
