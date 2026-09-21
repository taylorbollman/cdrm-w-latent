"""Independent FP64 adjudication of bounded tiled-versus-scan FP32 roundoff.

This diagnostic preserves the original elementwise FP32 screen.  It supplies
additional evidence, not an amended acceptance threshold or training clearance.
No GPU execution occurs at import; callers own the approved execution context.
"""
from __future__ import annotations

import copy
import math
import torch
from torch.nn import functional as F

from cdrm.pretrained.olmo_recurrent import recurrent_layer_reference
from cdrm.pretrained.olmo_tiled import tiled_recurrent_layer


def _fixed_native_rope_constants(positions, head_dim, base):
    # Preserve the finite positional matrix used by native FP32 code, while
    # removing FP32 rounding from its application to Q/K and from its VJP.
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    phase = positions.float().unsqueeze(-1) * inv
    phase = torch.cat((phase, phase), dim=-1).unsqueeze(1)
    return phase.cos().double(), phase.sin().double()


def _rotate64(x, cosine, sine):
    first, second = x.chunk(2, dim=-1)
    return x * cosine + torch.cat((-second, first), dim=-1) * sine


def fp64_block_oracle(layer, x, positions, valid, alpha):
    """Independent sequential FP64 block; frozen FP32-native parameter values.

    Persistent writes are unrotated and use LN((1-alpha)*x+alpha*z).  Historical
    keys/values are reconstructed as attached tensors; the current diagonal is
    temporary.  No O1 projection, RoPE, attention or finish helper is called.
    """
    if x.dtype != torch.float64:
        raise ValueError("The adjudication oracle requires float64 input")
    config = layer.config
    wq, wo, wf, wd = (p.detach().to(device=x.device, dtype=torch.float64) for p in
                      (layer.att_proj.weight, layer.attn_out.weight, layer.ff_proj.weight, layer.ff_out.weight))
    cosine, sine = _fixed_native_rope_constants(positions, config.head_dim, config.rope_freq_constant)
    def project(value):
        normalized = F.layer_norm(value, (config.model_dim,), eps=config.layer_norm_eps)
        chunks = F.linear(normalized, wq).split(config.model_dim, -1)
        return tuple(t.view(x.shape[0], value.shape[1], config.num_heads, config.head_dim).transpose(1, 2) for t in chunks)
    keys, values, outputs = [], [], []
    for index in range(x.shape[1]):
        current = x[:, index:index + 1]
        q, temporary_k, temporary_v = project(current)
        q = _rotate64(q, cosine[:, :, index:index + 1], sine[:, :, index:index + 1])
        history_k = torch.cat((*keys, temporary_k), dim=-2)
        history_v = torch.cat((*values, temporary_v), dim=-2)
        history_k = _rotate64(history_k, cosine[:, :, :index + 1], sine[:, :, :index + 1])
        score = q @ history_k.transpose(-1, -2) / math.sqrt(config.head_dim)
        allowed = valid[:, None, None, :index + 1]
        nonempty = allowed.any(-1, keepdim=True)
        score = score.masked_fill(~allowed, -torch.inf)
        probability = torch.where(nonempty, score, torch.zeros_like(score)).softmax(-1) * nonempty
        attended = (probability @ history_v).transpose(1, 2).reshape(current.shape)
        residual = current + F.linear(attended, wo)
        normalized = F.layer_norm(residual, (config.model_dim,), eps=config.layer_norm_eps)
        up, gate = F.linear(normalized, wf).chunk(2, dim=-1)
        completed = residual + F.linear(F.silu(gate) * up, wd)
        outputs.append(completed)
        _, key, value = project((1.0 - alpha) * current + alpha * completed)
        keys.append(key); values.append(value)
    return torch.cat(outputs, dim=1), (torch.cat(keys, dim=-2), torch.cat(values, dim=-2))


def _error(actual, expected):
    a, b = actual.detach().double(), expected.detach().double()
    difference = a - b
    norm, error = torch.linalg.vector_norm(b).item(), torch.linalg.vector_norm(difference).item()
    return {"finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "max_abs": difference.abs().max().item(), "difference_l2": error,
            "reference_l2": norm, "relative_l2": error / max(norm, 1e-300),
            "reference_max_abs": b.abs().max().item(),
            "tensor_relative_max": difference.abs().max().item() / max(b.abs().max().item(), 1e-300)}


def roundoff_check(layer, x, positions, valid, go, gk, gv, alpha):
    """Compare the same input/cotangents with two FP32 paths and FP64 arithmetic."""
    original_atol, original_rtol = 2e-6, 3e-4
    with torch.autocast(x.device.type, enabled=False):
        frozen = copy.deepcopy(layer).to(device=x.device, dtype=torch.float32).requires_grad_(False)
        gradients, outputs = {}, {}
        for name in ("tiled_fp32", "scan_fp32", "oracle_fp64"):
            probe = x.detach().to(torch.float64 if name == "oracle_fp64" else torch.float32).clone().requires_grad_()
            if name == "oracle_fp64":
                out, pair = fp64_block_oracle(frozen, probe, positions, valid, alpha)
            else:
                fn = tiled_recurrent_layer if name == "tiled_fp32" else recurrent_layer_reference
                options = {"attention_precision": "mixed"} if name == "tiled_fp32" else {"attention_backend": "math"}
                out, pair = fn(frozen, probe, alpha=alpha, past=None, query_positions=positions,
                               key_positions=positions, key_valid=valid, **options)
            gradients[name] = torch.autograd.grad((out, *pair), probe,
                                                  grad_outputs=tuple(g.to(probe.dtype) for g in (go, gk, gv)))[0].detach()
            outputs[name] = out.detach()
        tiled, scan, oracle = (gradients[k].double() for k in ("tiled_fp32", "scan_fp32", "oracle_fp64"))
        difference = (tiled - scan).abs()
        budget = original_atol + original_rtol * scan.abs()
        failures = difference > budget
        failing_indices = failures.flatten().nonzero().flatten()
        ranked = torch.argsort((difference / budget).flatten(), descending=True)
        chosen = ranked[:min(32, max(1, failing_indices.numel()))]
        rows = []
        for flat in chosen.tolist():
            b, rem = divmod(flat, x.shape[1] * x.shape[2]); t, d = divmod(rem, x.shape[2])
            index = (b, t, d)
            rows.append({"index": list(index), "original_failed": bool(failures[index]),
                         "tiled_fp32": tiled[index].item(), "scan_fp32": scan[index].item(),
                         "oracle_fp64": oracle[index].item(), "original_limit": budget[index].item(),
                         "fp32_pair_difference": difference[index].item(),
                         "tiled_error_vs_fp64": abs((tiled - oracle)[index].item()),
                         "scan_error_vs_fp64": abs((scan - oracle)[index].item())})
        return {"schema": "olmo-tiled-roundoff-diagnostic-v1", "alpha": alpha, "shape": list(x.shape),
                "scope": "Input-gradient adjudication with frozen identical native FP32 parameter values; no revised acceptance threshold",
                "oracle_arithmetic": "FP64 LN/projections/softmax/MLP/RoPE application; fixed native FP32 positional sine/cosine coefficients",
                "original_elementwise_screen": {"atol": original_atol, "rtol": original_rtol,
                                                "passed": not bool(failures.any()), "failure_count": int(failures.sum()),
                                                "coordinates": tiled.numel()},
                "input_gradient": {"tiled_vs_scan": _error(tiled, scan), "tiled_vs_fp64": _error(tiled, oracle),
                                   "scan_vs_fp64": _error(scan, oracle)},
                "output": {"tiled_vs_fp64": _error(outputs["tiled_fp32"], outputs["oracle_fp64"]),
                           "scan_vs_fp64": _error(outputs["scan_fp32"], outputs["oracle_fp64"])},
                "worst_budget_coordinates": rows, "reported_coordinate_limit": 32,
                "cotangent_l2": {name: torch.linalg.vector_norm(value.double()).item()
                                 for name, value in (("output", go), ("permanent_key", gk), ("permanent_value", gv))},
                "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32)}
