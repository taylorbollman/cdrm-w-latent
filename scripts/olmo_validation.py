"""Numerical/report helpers for bounded native OLMo validation.

The tensor-relative FP32 check requires both L2 and maximum-error bounds so a
small global gradient norm cannot hide a defective individual tensor. Initial
budgets are semantic screens, not BF16 training acceptance criteria.
"""
from __future__ import annotations
from contextlib import nullcontext
import math
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

PROMPTS = (
    "A program keeps a running total. It starts at zero, adds three, subtracts "
    "one, and then doubles the result. The final total is four. Explain each "
    "operation in order and check the calculation.",
    "def prefix_sums(values):\n    total = 0\n    result = []\n    for value in values:\n"
    "        total += value\n        result.append(total)\n    return result\n\n"
    "For the input [2, 3, -1], the output is [2, 5, 4].",
)


def require_container_gpu() -> dict:
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run with scripts/docker_shell.sh; host GPU execution is forbidden")
    if Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("The container working directory must be /workspace/cdrm-w-latent")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; there is no CPU fallback")
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "nvidia_smi": smi}


def comparison(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> dict:
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    a, b = actual.detach().float(), expected.detach().float()
    difference = a - b
    norm = float(torch.linalg.vector_norm(b))
    error = float(torch.linalg.vector_norm(difference))
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"passed": finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
            "finite": finite, "max_abs": float(difference.abs().max()),
            "relative_l2": error / max(norm, 1e-30), "reference_l2": norm,
            "difference_l2": error, "atol": atol, "rtol": rtol}


def compare_gradients(model, reference, *, atol: float, rtol: float) -> dict:
    actual = dict(model.named_parameters())
    expected = dict(reference.named_parameters())
    if actual.keys() != expected.keys():
        raise AssertionError("Native and adapter parameter ownership differ")
    rows, missing = {}, []
    squared_error = squared_reference = 0.0
    for name, parameter in actual.items():
        a, b = parameter.grad, expected[name].grad
        if a is None or b is None:
            missing.append(name)
            continue
        row = comparison(a, b, atol=atol, rtol=rtol)
        rows[name] = row
        squared_error += row["difference_l2"] ** 2
        squared_reference += row["reference_l2"] ** 2
    failed = [name for name, row in rows.items() if not row["passed"]]
    return {"passed": not missing and not failed, "missing": missing, "failed": failed,
            "parameter_tensors": len(actual), "tensors": rows,
            "relative_l2": math.sqrt(squared_error / max(squared_reference, 1e-60)),
            "max_abs": max((row["max_abs"] for row in rows.values()), default=0.0)}


def loss_of(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def autocast(precision: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16_mixed" else nullcontext()


def descriptive(row):
    """Remove acceptance labels from cross-precision/backend observations."""
    if isinstance(row, dict):
        return {k: descriptive(v) for k, v in row.items()
                if k not in ("passed", "atol", "rtol", "failed")}
    return row


def semantic_gradients(model, reference):
    """FP32 gradient budget accounting for cancellation and reduction order.

    Preserve the stricter elementwise result as evidence. Acceptance requires
    both a small tensor L2 error and a small error relative to that tensor's
    largest component. The absolute floor only handles nearly zero tensors.
    """
    result = compare_gradients(model, reference, atol=2e-6, rtol=3e-4)
    result["elementwise_failed"] = result.pop("failed")
    actual, expected = dict(model.named_parameters()), dict(reference.named_parameters())
    for name, row in result["tensors"].items():
        a, b = actual[name].grad.detach().float(), expected[name].grad.detach().float()
        difference = (a - b).abs()
        reference_max = float(b.abs().max())
        violations = difference > 2e-6 + 3e-4 * b.abs()
        row.update(elementwise_passed=row.pop("passed"),
                   elementwise_violation_count=int(violations.sum()), reference_max_abs=reference_max,
                   tensor_relative_max=difference.max().item() / max(reference_max, 1e-30),
                   maximum_error_limit=2e-6 + 1e-4 * reference_max,
                   relative_l2_limit=1e-4)
        # Avoid defining a relative norm failure for a nearly zero tensor only
        # when its *whole error norm* is below the same small absolute floor.
        row["passed"] = row["finite"] and row["max_abs"] <= row["maximum_error_limit"] and (
            row["relative_l2"] <= 1e-4 or row["difference_l2"] <= 2e-6)
        if row["elementwise_violation_count"]:
            index = int((difference - (2e-6 + 3e-4 * b.abs())).flatten().argmax())
            row["worst_elementwise_violation"] = {
                "flat_index": index, "actual": float(a.flatten()[index]),
                "reference": float(b.flatten()[index]), "absolute_error": float(difference.flatten()[index]),
            }
    result["failed"] = [name for name, row in result["tensors"].items() if not row["passed"]]
    result["passed"] = not result["missing"] and not result["failed"]
    return result


def snapshot(model, output, input_gradient, loss):
    return {
        "gradients": {name: p.grad.detach().cpu().clone()
                      for name, p in model.named_parameters() if p.grad is not None},
        "logits": output.logits.detach().cpu().clone(),
        "input_gradient": input_gradient.detach().cpu().clone(),
        "loss": float(loss.detach()),
    }


def compare_snapshot(model, output, input_gradient, loss, fp32):
    rows = {}
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or name not in fp32["gradients"]:
            missing.append(name)
            continue
        rows[name] = descriptive(comparison(parameter.grad.detach().cpu(), fp32["gradients"][name], atol=0, rtol=0))
    gradient = {
        "tensors": rows, "missing": missing, "parameter_tensors": len(rows),
        "relative_l2": math.sqrt(sum(row["difference_l2"] ** 2 for row in rows.values()) /
                                 max(sum(row["reference_l2"] ** 2 for row in rows.values()), 1e-60)),
        "max_abs": max(row["max_abs"] for row in rows.values()),
    }
    logits = output.logits.detach().cpu().float()
    return {
        "logits": descriptive(comparison(logits, fp32["logits"], atol=0, rtol=0)),
        "input_gradient": descriptive(comparison(input_gradient.detach().cpu(), fp32["input_gradient"], atol=0, rtol=0)),
        "gradients": gradient, "loss": float(loss.detach()), "fp32_loss": fp32["loss"],
        "top_token_disagreement_fraction": float((logits.argmax(-1) != fp32["logits"].argmax(-1)).float().mean()),
        "finite": not missing and all(row["finite"] for row in rows.values()) and bool(torch.isfinite(logits).all()),
        "scope": "descriptive BF16-versus-FP32 observation, not training acceptance",
    }
