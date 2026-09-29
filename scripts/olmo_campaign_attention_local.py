#!/usr/bin/env python3
"""Actual ordinary-attention inputs with fixed incoming CE cotangents.

Reproduce one pinned production CE anchor, observing eight sites: ordinary
layers 1/14, passes 0/3, both original fixture records. Observation excludes
checkpoint recomputation and does not replace attention math. Then release
the model and compare FP32 math, BF16 math and BF16 Flash on identical actual
Q/K/V values and incoming cotangents. No optimizer, DDP or CUDA graphs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained import olmo
from cdrm.pretrained.artifacts import sha256_file, write_json
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_aux_cotangents import _encode_tensor
from scripts.olmo_campaign_backend_cross import (
    TRITON, anchor_comparison, configure_cross_path, source_hashes as cross_sources,
)
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_bridge import capture_passes, tensor_geometry
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, component_backward, record_gradients
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

PATHS = ("fp32_math", "bf16_math", "bf16_flash")
NAMES = ("query", "key", "value")


def layout_record(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "storage_offset": tensor.storage_offset()}


def _snapshot(tensor):
    return tensor.detach().cpu().clone()


class _FunctionalProxy:
    def __init__(self, original, attention):
        self.original, self.attention = original, attention

    def __getattr__(self, name):
        return self.attention if name == "scaled_dot_product_attention" else getattr(self.original, name)


@contextmanager
def capture_attention(model, fixtures, *, layers=(1, 14), passes=(0, 3)):
    """Observe original forwards only, with transparent tensor-gradient hooks.

    Only the ``olmo.F`` module variable is temporarily replaced. The actual
    torch.nn.functional module, checkpoint implementation and attention call
    are unchanged. Outer FBT pre/return hooks bracket original forwards;
    non-reentrant checkpoint recomputation runs after that scope has closed.
    No live Q/K/V/output tensor is retained by the observation records.
    """
    batches = [batch for rows, _ in fixtures for batch in rows]
    observations = {"sites": [], "forward_calls": 0, "original_sdpa_calls": 0,
                    "excluded_recompute_sdpa_calls": 0}
    original, active = olmo.F, None
    target_layers, target_passes = tuple(layers), tuple(passes)
    if not target_layers or not target_passes or len(set(target_layers)) != len(target_layers) or len(set(target_passes)) != len(target_passes):
        raise ValueError("Distinct selected layers/passes are required")

    def enter(module, args, kwargs):
        nonlocal active
        if active is not None:
            raise AssertionError("Nested outer FBT forward is outside observer scope")
        mode = kwargs["mode"]
        if not mode.enabled or mode.first_pass_policy != "configured-rt-v1":
            raise ValueError("Observer requires finite configured-RT FBT passes")
        ordinary = tuple(i for i in range(len(module.backbone.layers)) if i not in mode.rt_mode.selected_layers)
        if not set(target_layers) <= set(ordinary) or any(p < 0 or p >= mode.num_passes for p in target_passes):
            raise ValueError("Selected site is outside this ordinary-layer/pass schedule")
        record = observations["forward_calls"]
        if record >= len(batches):
            raise AssertionError("Unexpected repeated outer FBT forward")
        observations["forward_calls"] += 1
        active = {"record": record, "ordinary": ordinary, "passes": mode.num_passes, "calls": 0}

    def leave(module, args, kwargs, output):
        nonlocal active
        state, active = active, None
        if output is not None and state is not None and state["calls"] != len(state["ordinary"])*state["passes"]:
            raise AssertionError("Observed ordinary SDPA call order/count differs")

    def attention(query, key, value, *args, **kwargs):
        if active is None:
            observations["excluded_recompute_sdpa_calls"] += 1
            return original.scaled_dot_product_attention(query, key, value, *args, **kwargs)
        index = active["calls"]
        active["calls"] += 1
        observations["original_sdpa_calls"] += 1
        pass_index, layer_offset = divmod(index, len(active["ordinary"]))
        layer = active["ordinary"][layer_offset]
        output = original.scaled_dot_product_attention(query, key, value, *args, **kwargs)
        if layer not in target_layers or pass_index not in target_passes:
            return output
        # Native full-sequence scope: preserve its exact causal invocation.
        if args or set(kwargs) - {"attn_mask", "dropout_p", "is_causal"}:
            raise ValueError("Observed attention arguments exceed the pinned native call")
        if kwargs.get("attn_mask") is not None or kwargs.get("dropout_p", 0.) != 0. or kwargs.get("is_causal") is not True:
            raise ValueError("Local check requires the actual mask-free causal dropout-zero call")
        tensors = dict(zip(NAMES, (query, key, value)))
        tensors["output"] = output
        row = {"key": f"record-{active['record']}/pass-{pass_index}/layer-{layer}",
            "record": active["record"], "pass": pass_index, "layer": layer,
            "valid_queries": _snapshot(batches[active["record"]].valid_mask),
            "tensors": {name: _snapshot(t) for name, t in tensors.items()},
            "layouts": {name: layout_record(t) for name, t in tensors.items()},
            "attention": {"attn_mask": None, "dropout_p": 0., "is_causal": True,
                          "scale": None, "enable_gqa": False}, "cotangent_calls": 0}
        observations["sites"].append(row)

        def incoming(gradient):
            if row["cotangent_calls"]:
                raise AssertionError("Attention output cotangent observed more than once")
            row["cotangent_calls"] += 1
            row["tensors"]["cotangent"] = _snapshot(gradient)
            row["layouts"]["cotangent"] = layout_record(gradient)
            # Return None: do not replace, cast, normalize or scale the cotangent.

        output.register_hook(incoming)
        return output

    pre = model.backbone.register_forward_pre_hook(enter, with_kwargs=True)
    post = model.backbone.register_forward_hook(leave, with_kwargs=True, always_call=True)
    olmo.F = _FunctionalProxy(original, attention)
    try:
        yield observations
    finally:
        olmo.F = original
        pre.remove()
        post.remove()
        active = None
    expected = len(batches)*len(target_layers)*len(target_passes)
    if observations["forward_calls"] != len(batches) or len(observations["sites"]) != expected:
        raise AssertionError("Missing selected attention site")
    if any(row["cotangent_calls"] != 1 for row in observations["sites"]):
        raise AssertionError("Missing actual attention-output cotangent")


def fixed_leaf(value, layout, *, device, dtype, requires_grad):
    """Independent leaf with actual element strides; no input alias promise.

    Gapped fused-projection V views can be densified by clone(preserve_format).
    empty_strided preserves those strides explicitly. Storage offset starts at
    zero; original offset is retained as evidence, not a functional dependency.
    """
    if tuple(value.shape) != tuple(layout["shape"]) or any(s < 0 for s in layout["stride"]):
        raise ValueError("Captured tensor shape/strides differ")
    result = torch.empty_strided(value.shape, tuple(layout["stride"]), dtype=dtype, device=device)
    with torch.no_grad():
        result.copy_(value)
    return result.requires_grad_(requires_grad)


def local_vjp(site, *, path, device):
    if path not in PATHS:
        raise ValueError("Unknown local precision/backend")
    device = torch.device(device)
    if path == "bf16_flash" and device.type != "cuda":
        raise ValueError("Flash local check requires CUDA; no CPU fallback")
    if any(site["tensors"][name].dtype != torch.bfloat16 for name in (*NAMES, "cotangent")):
        raise ValueError("Actual attention anchor must have BF16 Q/K/V and incoming cotangent")
    dtype = torch.float32 if path == "fp32_math" else torch.bfloat16
    qkv = tuple(fixed_leaf(site["tensors"][name], site["layouts"][name], device=device,
                           dtype=dtype, requires_grad=True) for name in NAMES)
    cotangent = fixed_leaf(site["tensors"]["cotangent"], site["layouts"]["cotangent"],
                           device=device, dtype=dtype, requires_grad=False)
    backend = SDPBackend.FLASH_ATTENTION if path == "bf16_flash" else SDPBackend.MATH
    with torch.autocast(device.type, enabled=False), sdpa_kernel(backend):
        output = torch.nn.functional.scaled_dot_product_attention(*qkv, **site["attention"])
        gradients = torch.autograd.grad(output, qkv, grad_outputs=cotangent)
    values = {"output": _snapshot(output), **{name: _snapshot(g) for name, g in zip(NAMES, gradients)}}
    input_digests, cotangent_digest = tree_digests(dict(zip(NAMES, qkv))), tree_digests(cotangent)
    common_values = {name: input_digests[name] == tree_digests(site["tensors"][name].to(dtype=dtype)) for name in NAMES}
    common_values["cotangent"] = cotangent_digest == tree_digests(site["tensors"]["cotangent"].to(dtype=dtype))
    return {"values": values,
        "input_digests": input_digests, "cotangent_digest": cotangent_digest,
        "common_value_checks": common_values,
        "input_layouts": {name: layout_record(t) for name, t in zip(NAMES, qkv)},
        "cotangent_layout": layout_record(cotangent), "output_layout": layout_record(output),
        "finite": all(bool(torch.isfinite(v).all()) for v in values.values()),
        "copied_inputs": "Independent leaves; native strides preserved; offset zero; Q/K/V aliasing not preserved"}


@torch.no_grad()
def attention_context(site):
    """FP32 causal-logit/attention summaries on valid queries, no thresholds."""
    q, k = (site["tensors"][name].float() for name in ("query", "key"))
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    allowed = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
    selected = site["valid_queries"][:, None, :].expand(scores.shape[:-1])
    high = scores.masked_fill(~allowed, -torch.inf).amax(-1)[selected]
    low = scores.masked_fill(~allowed, torch.inf).amin(-1)[selected]
    maximum = scores.masked_fill(~allowed, -torch.inf).softmax(-1).amax(-1)[selected]
    summary = lambda t: {"min": float(t.min()), "mean": float(t.mean()), "max": float(t.max())}
    return {"valid_query_heads": int(selected.sum()), "logit_row_range": summary(high-low),
            "max_attention_probability": summary(maximum), "logit_row_max": summary(high),
            "logit_row_min": summary(low), "scope": "FP32 qk/sqrt(d) on captured BF16 values, causal keys and actual valid queries; descriptive only"}


def source_hashes():
    sources = cross_sources()
    for name in ("scripts/olmo_campaign_attention_local.py",
                 "docs/reports/olmo-precision-localization/attention-local-protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def load_reference(path, digest, sources):
    path = Path(path)
    if path.stat().st_size > 64*1024*1024 or sha256_file(path) != digest:
        raise ValueError("Pinned crossed-backend reference bytes differ")
    reference = json.loads(path.read_text())
    if (sha256_file(path) != digest or reference.get("schema") != "olmo-campaign-backend-cross-v1"
            or reference.get("status") != "passed_operational_diagnostic" or reference.get("passed") is not True
            or not reference.get("integrity") or not all(reference["integrity"].values())):
        raise ValueError("Crossed-backend reference is incomplete or changed")
    if not reference.get("sources") or any(sources.get(k) != v for k, v in reference["sources"].items()):
        raise ValueError("Crossed-backend source pins differ")
    anchors = [row for row in reference["rows"] if row["path"] == TRITON]
    if len(anchors) != 1 or not anchors[0]["passed"]:
        raise ValueError("One successful production CE anchor required")
    return reference, anchors[0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.reference_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.reference_sha256):
        parser.error("Reference SHA256 must be 64 lowercase hexadecimal characters")
    return args


def main(argv=None):
    args = parse_args(argv)
    deterministic = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Local attention diagnostic requires one process without DDP")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "olmo-campaign-attention-local-v1", "status": "running", "passed": False,
        "scope": __doc__, "sources": source_hashes(), "runtime": runtime, "determinism": deterministic,
        "started_utc": datetime.now(timezone.utc).isoformat(), "reference_sha256": args.reference_sha256,
        "rows": [], "paths": PATHS, "selected_layers": [1, 14], "selected_passes": [0, 3],
        "aggregate_anchor_backwards": 1, "physical_anchor_backwards": 2, "local_vjps": 24,
        "optimizer_updates": 0, "qualification": "No new numerical acceptance threshold; earlier qualifications remain",
        "geometry_scope": "Output/dQ/dK/dV geometry uses all call entries; valid_query_output separately selects actual valid queries. Incoming cotangent is fixed from the production CE backward.",
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for name in report["sources"]:
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-precision-localization", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    def publish(stage, row):
        report["rows"].append({"stage": stage, **row})
        persist(stage)
        tracker.log(scalar_metrics(row, "diagnostic/"+stage), step=len(report["rows"]))
        print({"stage": stage, "passed": row["passed"], "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
        if not row["passed"]:
            raise AssertionError("Attention-local operational/anchor gate failed")

    try:
        tracker.start({k: report[k] for k in ("scope", "paths", "determinism", "reference_sha256", "qualification")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("load_reference_and_source")
        previous, anchor = load_reference(args.reference_report, args.reference_sha256, report["sources"])
        model, recipe, checkpoint, ids, eos = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NFR", torch.device("cuda"))
        original = {key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS}
        fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0, length=16,
                    token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
        inputs = tree_digests([{"batches": [vars(b) for b in batches], "noise": noises} for batches, noises in fixtures])
        contracts = {"runtime": runtime == previous["runtime"], "checkpoint": checkpoint == previous["source_checkpoint"],
            "recipe": recipe.to_dict() == previous["recipe"] and recipe.sha256 == previous["recipe_sha256"],
            "fixture_inputs": inputs == previous["fixture_inputs"], "runtime_flags": original == previous["production_runtime_flags"],
            "determinism": deterministic == previous["determinism"],
            "math_reduction": report["math_sdpa_reduced_precision_reduction"] == previous["math_sdpa_reduced_precision_reduction"],
            "bf16_matmul_reduction": report["bf16_matmul_reduced_precision_reduction"] == previous["bf16_matmul_reduced_precision_reduction"]}
        report.update(source_checkpoint=checkpoint, recipe=recipe.to_dict(), recipe_sha256=recipe.sha256,
                      fixture_inputs=inputs, reference_contract_checks=contracts)
        if not all(contracts.values()):
            raise AssertionError("Production anchor contract differs")
        weights, rng = tree_digests(dict(model.named_parameters())), rng_snapshot()
        execution = configure_cross_path(model, original, TRITON)
        persist("production_ce_anchor_with_observer")
        with capture_attention(model, fixtures) as observed, capture_passes(model, fixtures) as passes, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            metrics = component_backward(model, recipe, fixtures, precision="bf16_mixed", layout="sparse", objective="ce")
        gradients, _ = record_gradients(model, scope="Production CE observer anchor; no new gradient budget")
        fingerprints = tree_digests([{k: row[k] for k in ("batch", "token_embeddings", "pass_hidden_states")} for row in passes])
        comparison = anchor_comparison(metrics, fingerprints, gradients,
            {**anchor, "gradients_vs_fp32": anchor["gradients_vs_bf16_math_eager"]})
        weights_equal = weights == tree_digests(dict(model.named_parameters()))
        inputs_equal = inputs == tree_digests([{"batches": [vars(b) for b in batches], "noise": noises} for batches, noises in fixtures])
        publish("production_ce_anchor_exact", {"execution": execution, "metrics": metrics, "gradients": gradients,
            "anchor_comparison": comparison, "observer": {k: v for k, v in observed.items() if k != "sites"},
            "weights_unchanged": weights_equal, "inputs_unchanged": inputs_equal, "rng_unchanged": rng_unchanged(rng),
            "passed": comparison["passed"] and weights_equal and inputs_equal and rng_unchanged(rng) and gradients["finite"]})
        captured_before = tree_digests(observed["sites"])
        payload = {"schema": "olmo-attention-local-fixture-v1", "reference_sha256": args.reference_sha256,
                   "sources": report["sources"], "sites": [{**{k: v for k, v in site.items() if k not in ("tensors", "valid_queries")},
                       "valid_queries": _encode_tensor(site["valid_queries"]),
                       "tensors": {k: _encode_tensor(v) for k, v in site["tensors"].items()}} for site in observed["sites"]]}
        fixture_path = args.output_dir/"attention-fixture.json"
        write_json(fixture_path, payload)
        fixture_sha = sha256_file(fixture_path)
        report["attention_fixture"] = {"path": str(fixture_path), "sha256": fixture_sha,
                                       "size_bytes": fixture_path.stat().st_size, "sites": len(observed["sites"])}
        model.zero_grad(set_to_none=True)
        del model, passes
        gc.collect()
        torch.cuda.empty_cache()
        for site in observed["sites"]:
            references = {}
            context = attention_context(site)
            for path in PATHS:
                persist(site["key"]+"/"+path)
                result = local_vjp(site, path=path, device="cuda")
                values = result.pop("values")
                row = {"site": site["key"], "path": path, "invocation": site["attention"],
                    "captured_layouts": site["layouts"], "captured_digests": tree_digests(site["tensors"]),
                    "local": result, "values": tree_digests(values), "attention_context": context,
                    "geometry": {k: tensor_geometry(v) for k, v in values.items()}, "memory": memory(),
                    "passed": result["finite"] and all(result["common_value_checks"].values()) and rng_unchanged(rng)}
                for reference_name, reference in references.items():
                    row["versus_"+reference_name] = {k: tensor_geometry(v, reference[k]) for k, v in values.items()}
                if path == "bf16_flash":
                    row["production_output_bitwise_exact"] = tree_digests(values["output"]) == tree_digests(site["tensors"]["output"])
                    row["passed"] &= row["production_output_bitwise_exact"]
                valid = site["valid_queries"][:, None, :, None].expand_as(values["output"])
                row["valid_query_output"] = tensor_geometry(values["output"][valid],
                    None if not references else references["fp32_math"]["output"][valid])
                references[path] = values
                publish(site["key"]+"/"+path, row)
        inputs_equal = inputs == tree_digests([{"batches": [vars(b) for b in batches], "noise": noises} for batches, noises in fixtures])
        captured_equal = captured_before == tree_digests(observed["sites"])
        publish("fixed_source_fixture_rng_integrity", {"sources_unchanged": report["sources"] == source_hashes(),
            "reference_unchanged": args.reference_sha256 == sha256_file(args.reference_report),
            "fixture_unchanged": fixture_sha == sha256_file(fixture_path), "rng_unchanged": rng_unchanged(rng),
            "original_inputs_unchanged": inputs_equal, "captured_sites_unchanged": captured_equal,
            "passed": report["sources"] == source_hashes() and args.reference_sha256 == sha256_file(args.reference_report)
                and fixture_sha == sha256_file(fixture_path) and inputs_equal and captured_equal and rng_unchanged(rng)})
        report.update(status="passed_operational_diagnostic", passed=True)
        persist("complete")
    except BaseException as error:
        failure = error
        report.update(status="failed", passed=False,
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist(report.get("stage", "setup"))
        try:
            tracker.finish(succeeded=report["passed"])
        except BaseException as error:
            report.update(status="failed", passed=False, tracking_finish_error={"type": type(error).__name__})
            if failure is None:
                raise
        finally:
            persist(report.get("stage", "setup"))


if __name__ == "__main__":
    main()
