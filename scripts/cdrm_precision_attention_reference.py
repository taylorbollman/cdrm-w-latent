#!/usr/bin/env python3
"""Independent same-operand attention VJP for retained ordinary-block probes.

The analytic FP64 derivative evaluates the captured primal operands and fixed
incoming cotangent. It is not a finite difference through a quantized program.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm_tiled_validate import cpu, digest, preserve_rng, save_json
from experiment_tracking import OnlineTracker

ATOL, RTOL = 2e-6, 2e-5
REFERENCE_CONTRACT_SHA256 = "6660b964a2f5a678d4324ac2799c814ad617ce9acceabbed5e0b0e23101714cc"


def analytic_attention(q, k, v, mask, dy):
    """Independent analytic reverse pass; no autograd and no output rounding."""
    with torch.no_grad(), torch.autocast(q.device.type, enabled=False):
        q, k, v, dy = [x.double() for x in (q, k, v, dy)]
        score = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if mask is not None:
            score = score + mask.double()
        shifted = score - score.amax(dim=-1, keepdim=True)
        probabilities = shifted.exp()
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        output = probabilities @ v
        dp = dy @ v.transpose(-1, -2)
        ds = probabilities * (dp - (dp * probabilities).sum(dim=-1, keepdim=True))
        return {
            "output": output,
            "q_gradient": (ds @ k) / math.sqrt(q.shape[-1]),
            "k_gradient": (ds.transpose(-1, -2) @ q) / math.sqrt(q.shape[-1]),
            "v_gradient": probabilities.transpose(-1, -2) @ dy,
        }


def native_fp32_attention(q, k, v, mask, dy):
    """Independent primitive execution with the same representable operands."""
    q, k, v = [x.detach().float().requires_grad_() for x in (q, k, v)]
    with torch.autocast(q.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if mask is None else mask.float(), dropout_p=0., is_causal=False)
        gradients = torch.autograd.grad(output, (q, k, v), dy.float())
    return {"output": output.detach(), **dict(zip(
        ("q_gradient", "k_gradient", "v_gradient"), (x.detach() for x in gradients)))}


def describe(reference, actual):
    r, a = reference.detach().double(), actual.detach().double()
    if not bool(torch.isfinite(r).all() and torch.isfinite(a).all()):
        return {'finite': False, 'reference_nonfinite': int((~torch.isfinite(r)).sum()),
                'actual_nonfinite': int((~torch.isfinite(a)).sum()), 'elements': r.numel(),
                'reference_l2': None, 'error_l2': None, 'relative_l2': None,
                'maximum_absolute_error': None, 'mean_signed_error': None, 'exact': False}
    error = a - r
    return {
        "finite": bool(torch.isfinite(r).all() and torch.isfinite(a).all()),
        "reference_l2": float(r.norm()), "error_l2": float(error.norm()),
        "relative_l2": float(error.norm() / r.norm().clamp_min(1e-12)),
        "maximum_absolute_error": float(error.abs().max()),
        "mean_signed_error": float(error.mean()), "elements": r.numel(),
        "exact": bool(torch.equal(r, a)),
    }


def quantized_interval(reference, actual):
    """Prospective storage-aware FP32-floor interval; retained as engineering evidence."""
    r, a = reference.detach().double(), actual.detach()
    result = describe(r, a)
    if not result['finite']:
        result.update(storage_dtype=str(a.dtype), interval_pass=False,
                      interval_failures=int((~torch.isfinite(r) | ~torch.isfinite(a)).sum()))
        return result
    floor = ATOL + RTOL * r.abs()
    lower = (r - floor).to(a.dtype).double()
    upper = (r + floor).to(a.dtype).double()
    ad = a.double()
    outside = (ad < lower) | (ad > upper) | ~torch.isfinite(ad)
    excess = torch.maximum(lower - ad, ad - upper).clamp_min(0.)
    center = r.to(a.dtype)
    above = torch.nextafter(center, torch.full_like(center, float('inf'))).double()
    below = torch.nextafter(center, torch.full_like(center, -float('inf'))).double()
    spacing = torch.maximum((above-center.double()).abs(), (center.double()-below).abs()).clamp_min(1e-300)
    ulps = (ad-center.double()).abs()/spacing
    result.update(storage_dtype=str(a.dtype), interval_pass=not bool(outside.any()),
                  interval_failures=int(outside.sum()), interval_failure_fraction=float(outside.double().mean()),
                  maximum_interval_excess=float(excess.max()),
                  exact_once_rounded=bool(torch.equal(r.to(a.dtype), a)),
                  maximum_ulp_distance_from_once_rounded=float(ulps.max()),
                  mean_ulp_distance_from_once_rounded=float(ulps.mean()),
                  once_rounded_difference=describe(r.to(a.dtype), a),
                  interval="Q_dtype(reference64 +/- (2e-6 + 2e-5*abs(reference64)))")
    return result


def validate_capture(capture, expected_shape=None):
    """Reject semantics the independent formula does not represent."""
    if capture.get('dropout_p') != 0. or capture.get('is_causal') is not False:
        raise ValueError('Require zero dropout and an explicit additive causal mask')
    shape = tuple(capture['q'].shape)
    if len(shape) != 4 or (expected_shape is not None and shape != expected_shape):
        raise ValueError('Unexpected physical attention shape')
    for key in ('q', 'k', 'v', 'output', 'output_gradient', 'q_gradient', 'k_gradient', 'v_gradient'):
        value = capture.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Missing, nonfinite or incompatible attention tensor: {key}')
    for key in ('mask', 'bias_before_cast'):
        value = capture.get(key)
        if (not torch.is_tensor(value) or not value.is_floating_point() or value.ndim != 4
                or value.shape[0] not in (1, shape[0]) or value.shape[1] not in (1, shape[1])
                or tuple(value.shape[-2:]) != (shape[-2], shape[-2])
                or bool(torch.isnan(value).any() or torch.isposinf(value).any())):
            raise ValueError(f'Unsupported additive attention bias: {key}')
    future = torch.ones(shape[-2], shape[-2], dtype=torch.bool).triu(1)
    if not bool((capture['mask'][..., future] < -1e20).all()):
        raise ValueError('Captured attention mask does not exclude every future token')


def original_bias_with_same_causality(actual_mask, original_bias, length):
    """Change finite positional bias only; preserve the actual future exclusion."""
    original = original_bias[..., :length, :length].float().clone()
    future = torch.ones((length, length), dtype=torch.bool, device=original.device).triu(1)
    # The original model reapplies this causal exclusion after its dtype cast.
    return original.masked_fill(future, torch.finfo(torch.float32).min)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--contract", type=Path, default=Path("docs/reports/cdrm-numerical-resolution/reference-contract.md"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-run-name", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new numerical diagnostic directory")
    if digest(args.contract) != REFERENCE_CONTRACT_SHA256:
        raise ValueError("The prospective local reference contract has changed")
    from cdrm_tiled_common import setup, sources, snapshot_sources, verify_sources
    runtime = setup(0)
    args.output_dir.mkdir(parents=True)
    started, tracker, retained = time.monotonic(), None, {}
    report = {"schema": "cdrm-same-operand-attention-v1", "status": "running",
              "scope": "Saved ordinary-attention operands; analytic local derivatives and positional-bias intervention, no training or numerical clearance",
              "numerical_clearance": False, "runtime": runtime,
              "contract": {"path": str(args.contract), "sha256": digest(args.contract)},
              "source": {"path": __file__, "sha256": digest(__file__)},
              "probe": {"path": str(args.probe_dir),
                        "report_sha256": digest(args.probe_dir / "report.json"),
                        "tensors_sha256": digest(args.probe_dir / "tensors.pt")},
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "observations": {}}
    try:
        tracked = sources((Path(__file__).resolve().relative_to(Path.cwd()),
                           Path('scripts/cdrm_tiled_validate.py'),
                           Path('tests/test_cdrm_precision_attention_reference.py'),
                           args.contract))
        report['source_sha256'] = tracked
        snapshot_sources(args.output_dir, tracked)
        (args.output_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        (args.output_dir / "reference-contract.md").write_bytes(args.contract.read_bytes())
        probe_report = json.loads((args.probe_dir/'report.json').read_text())
        artifact = probe_report.get('tensor_artifact', probe_report.get('tensors', {}))
        if probe_report['status'] != 'diagnostics_complete' or artifact.get('sha256') != report['probe']['tensors_sha256']:
            raise ValueError('Require a completed probe with its exact declared tensor artifact')
        data = torch.load(args.probe_dir / "tensors.pt", map_location="cpu", weights_only=False)
        tracker = OnlineTracker(project="cdrm-numerical-resolution", entity="taylorbollman",
                                group="20260907T232931Z", name=args.wandb_run_name,
                                output_dir=args.output_dir, preserve_state=preserve_rng)
        report["wandb"] = tracker.record
        tracker.start({k: report[k] for k in ("scope", "contract", "source", "probe", "arguments")})
        for arm in args.arm:
            for block in args.blocks:
                name = f"{arm}/block{block}"
                capture = data["captures"][arm]["blocks"][str(block)]["sdpa"]
                validate_capture(capture, (64, 16, 256, 8))
                operands = [capture[key].cuda() for key in ("q", "k", "v", "mask", "output_gradient")]
                q, k, v, mask, dy = operands
                if tuple(q.shape[:3]) != (64, 16, 256) or q.shape[-1] != 8:
                    raise AssertionError("Primary local reference requires original full B64/H16/T256/Dhead8")
                reference = analytic_attention(q, k, v, mask, dy)
                same_fp32 = native_fp32_attention(q, k, v, mask, dy)
                original_bias = original_bias_with_same_causality(mask, capture["bias_before_cast"].cuda(), q.shape[-2])
                bias_control = analytic_attention(q, k, v, original_bias, dy)
                obs = {
                    "native_against_analytic": {key: quantized_interval(reference[key], capture[key].cuda()) for key in reference},
                    "local_fp32_against_analytic": {key: quantized_interval(reference[key], same_fp32[key]) for key in reference},
                    "fp32_bias_intervention_same_qkv_dy": {key: describe(reference[key], bias_control[key]) for key in reference},
                    "dtypes": {key: str(capture[key].dtype) for key in ("q", "k", "v", "mask", "bias_before_cast", "output", "output_gradient")},
                    "interpretation": "Local same-operand derivative checks; FP32-bias intervention changes only positional-bias precision at fixed QKV and incoming cotangent. It does not measure end-to-end gradient improvement."}
                valid = torch.ones_like(mask, dtype=torch.bool).tril()
                obs["unmasked_bias_difference"] = describe(original_bias.expand_as(mask)[valid], mask[valid])
                obs["reference_floor_supported"] = all(row["interval_pass"] for row in obs["local_fp32_against_analytic"].values())
                obs["native_local_interval_pass"] = all(row["interval_pass"] for row in obs["native_against_analytic"].values())
                report["observations"][name] = obs
                retained[name] = {"analytic": cpu(reference), "local_fp32": cpu(same_fp32), "fp32_bias": cpu(bias_control)}
                tracker.log({"block": block, f"local/{arm}/native_interval_pass": obs["native_local_interval_pass"],
                             f"local/{arm}/fp32_interval_pass": obs["reference_floor_supported"],
                             **{f"local/{arm}/{key}/relative_l2": value["relative_l2"] for key, value in obs["native_against_analytic"].items()}})
                print(json.dumps({"case": name, "native_interval_pass": obs["native_local_interval_pass"],
                                  "fp32_interval_pass": obs["reference_floor_supported"],
                                  "native_relative_l2": {key: value["relative_l2"] for key, value in obs["native_against_analytic"].items()},
                                  "bias_only_relative_l2": {key: value["relative_l2"] for key, value in obs["fp32_bias_intervention_same_qkv_dy"].items()}}), flush=True)
                del operands, q, k, v, mask, dy, reference, same_fp32, bias_control, original_bias
        if digest(args.contract) != report["contract"]["sha256"] or digest(__file__) != report["source"]["sha256"]:
            raise RuntimeError("Contract or reference source changed during execution")
        verify_sources(tracked)
        if digest(args.probe_dir/'report.json') != report['probe']['report_sha256'] or digest(args.probe_dir/'tensors.pt') != report['probe']['tensors_sha256']:
            raise RuntimeError('Probe inputs changed during reference execution')
        report["status"] = "diagnostics_complete"
        tracker.summary({"observations": len(report["observations"]), "numerical_clearance": False,
                         "all_local_fp32_intervals_pass": all(x["reference_floor_supported"] for x in report["observations"].values()),
                         "all_native_intervals_pass": all(x["native_local_interval_pass"] for x in report["observations"].values())})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            if retained:
                torch.save(retained, args.output_dir / "references.pt")
                report["tensor_artifact"] = {"path": str(args.output_dir / "references.pt"), "sha256": digest(args.output_dir / "references.pt")}
            if tracker:
                tracker.finish(succeeded=report["status"] == "diagnostics_complete")
        except Exception as error:
            report.update(status='failed', tracking_final_sync_error_type=type(error).__name__)
            raise
        finally:
            save_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
