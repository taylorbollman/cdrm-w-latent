#!/usr/bin/env python3
"""Fixed NF fusion/ordinary-stack boundaries with a common production cotangent.

Two exact NF CE anchors (four physical backwards) capture record0 transitions
into passes1/3. Twelve local VJPs separate inherited input perturbations from
module-level precision/backend differences. No optimizer, DDP, CUDA graphs,
policy change or additive attribution of whole-model gradient errors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
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

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.document_policy import feedback_eligibility
from cdrm.pretrained.recurrent import RTMode
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_attention_local import fixed_leaf, layout_record, source_hashes as attention_sources
from scripts.olmo_campaign_aux_cotangents import _encode_tensor
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_fusion_precision import (FP32, BF16, anchor_comparison, load_reference,
    reference_anchor, state_contract_checks, source_hashes as fusion_sources)
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_bridge import capture_passes, configure_path, forward_geometry, tensor_geometry
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, component_backward, gradient_geometry, record_gradients
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_recurrence_precision import arm_contract, fixture_pins, state_pins
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

PATHS = ("A_fp32_at_fp32_origin", "B_fp32_at_bf16_origin", "C_bf16_at_bf16_origin")
PASSES = (1, 3)


@contextmanager
def capture_boundaries(model, fixtures):
    """Transparent module/output hooks, active only inside original FBT forwards.

Fusion inputs already include shifting/jitter; its raw output precedes masking.
Stack inputs are the actual post-mixing/concatenation tensors, and its public
output includes final normalization/padding. Checkpoint replays of inner
ordinary layers are counted separately and never become new boundary sites.
"""
    batches = [batch for rows, _ in fixtures for batch in rows]
    observed = {"sites": {}, "outer_forwards": 0, "original_stack_calls": 0, "original_fusion_calls": 0,
                "original_layer_entries": 0, "excluded_recompute_layer_entries": 0}
    active, handles, tensor_hooks = None, [], []

    def enter(module, args, kwargs):
        nonlocal active
        mode = kwargs["mode"]
        if (active is not None or not mode.enabled or mode.num_passes != 4 or mode.rt_mode.selected_layers
                or mode.first_pass_policy != "configured-rt-v1" or not kwargs.get("right_padded_causal")):
            raise ValueError("Boundary observer requires original NF K4 right-padded forwards")
        record = observed["outer_forwards"]
        if record >= len(batches):
            raise AssertionError("Unexpected repeated outer FBT forward")
        observed["outer_forwards"] += 1
        active = {"record": record, "stack": 0, "fusion": 0, "mode": mode}

    def leave(module, args, kwargs, output):
        nonlocal active
        state, active = active, None
        if output is not None and (state is None or state["stack"] != 4 or state["fusion"] != 3):
            raise AssertionError("NF boundary invocation order/count differs")

    def save(kind, pass_index, inputs, kwargs, tensor_kwargs, output):
        if active["record"] != 0 or pass_index not in PASSES:
            return
        batch = batches[0]
        valid = (feedback_eligibility(batch.valid_mask, batch.document_ids, active["mode"].document_policy)
                 if kind == "fusion" else batch.valid_mask)
        tensors = {**inputs, **tensor_kwargs, "output": output, "valid_output_mask": valid}
        if any(t.dtype != torch.float32 for t in (*inputs.values(), output)):
            raise ValueError("Observed boundary inputs/output must actually be FP32; no promotion")
        key = f"record-0/pass-{pass_index}/{kind}"
        if key in observed["sites"]:
            raise AssertionError("Duplicate boundary capture")
        site = {"key": key, "record": 0, "pass": pass_index, "kind": kind,
                "input_names": list(inputs), "tensor_kwarg_names": list(tensor_kwargs), "kwargs": kwargs,
                "tensors": {name: value.detach().cpu().clone() for name, value in tensors.items()},
                "layouts": {name: layout_record(value) for name, value in tensors.items()}, "cotangent_calls": 0}
        observed["sites"][key] = site
        output_shape = tuple(output.shape)

        def incoming(gradient):
            if site["cotangent_calls"] or gradient.dtype != torch.float32 or tuple(gradient.shape) != output_shape:
                raise AssertionError("Boundary cotangent must arrive exactly once with original FP32 shape")
            site["cotangent_calls"] += 1
            site["tensors"]["cotangent"] = gradient.detach().cpu().clone()
            site["layouts"]["cotangent"] = layout_record(gradient)
            # Return None: never replace/scale the actual incoming gradient.

        tensor_hooks.append(output.register_hook(incoming))

    def fusion(module, args, kwargs, output):
        if active is None:
            raise AssertionError("Fusion unexpectedly replayed outside original FBT forward")
        index = active["fusion"] + 1
        if len(args) != 2 or kwargs or active["stack"] != index:
            raise AssertionError("Unexpected fusion invocation")
        active["fusion"] = index
        observed["original_fusion_calls"] += 1
        save("fusion", index, dict(zip(("previous_hidden", "token_input"), args)), {}, {}, output)

    def stack(module, args, kwargs, output):
        if active is None:
            raise AssertionError("Full stack unexpectedly replayed outside original FBT forward")
        index = active["stack"]
        required = {"inputs_embeds", "return_logits", "attention_mask", "position_ids", "right_padded_causal", "mode"}
        if (args or set(kwargs) != required or active["fusion"] != index
                or kwargs["return_logits"] is not False or kwargs["right_padded_causal"] is not True
                or kwargs["mode"].selected_layers):
            raise AssertionError("Unexpected full ordinary-stack invocation")
        active["stack"] += 1
        observed["original_stack_calls"] += 1
        save("stack", index, {"inputs_embeds": kwargs["inputs_embeds"]},
             {"return_logits": kwargs["return_logits"], "right_padded_causal": kwargs["right_padded_causal"],
              "mode": asdict(kwargs["mode"])},
             {name: kwargs[name] for name in ("attention_mask", "position_ids")}, output.last_hidden_state)

    def layer_enter(module, args):
        key = "original_layer_entries" if active is not None else "excluded_recompute_layer_entries"
        observed[key] += 1

    try:
        handles.append(model.backbone.register_forward_pre_hook(enter, with_kwargs=True))
        handles.append(model.backbone.register_forward_hook(leave, with_kwargs=True, always_call=True))
        handles.append(model.backbone.fusion.register_forward_hook(fusion, with_kwargs=True))
        handles.append(model.backbone.backbone.register_forward_hook(stack, with_kwargs=True))
        handles.extend(layer.register_forward_pre_hook(layer_enter) for layer in model.backbone.backbone.layers)
        yield observed
    finally:
        for handle in (*handles, *tensor_hooks):
            handle.remove()
        active = None
    if (observed["outer_forwards"] != len(batches) or observed["original_stack_calls"] != 4*len(batches)
            or observed["original_fusion_calls"] != 3*len(batches) or len(observed["sites"]) != 4
            or any(site["cotangent_calls"] != 1 for site in observed["sites"].values())):
        raise AssertionError("Incomplete original boundary/cotangent coverage")


def match_sites(fp32, bf16):
    if fp32.keys() != bf16.keys() or len(fp32) != 4:
        raise ValueError("Both precision anchors must contain the same four sites")
    for key, a in fp32.items():
        b = bf16[key]
        for field in ("key", "record", "pass", "kind", "input_names", "tensor_kwarg_names", "kwargs"):
            if tree_digests(a[field]) != tree_digests(b[field]):
                raise ValueError("Boundary invocation metadata differs between origins")
        for name in (*a["tensor_kwarg_names"], "valid_output_mask"):
            if tree_digests(a["tensors"][name]) != tree_digests(b["tensors"][name]):
                raise ValueError("Boundary masks/positions differ between origins")
        for name in (*a["input_names"], "output", "cotangent"):
            if a["tensors"][name].shape != b["tensors"][name].shape:
                raise ValueError("Boundary tensor shapes differ between origins")
        for name in a["input_names"]:
            if any(a["layouts"][name][field] != b["layouts"][name][field] for field in ("shape", "stride", "dtype")):
                raise ValueError("Boundary input dtype/shape/strides differ between origins")


def write_fixture(path, origins, *, provenance):
    """Self-contained small JSON/base64 capture; no parameter-gradient payloads."""
    encoded_origins = {}
    for origin, sites in origins.items():
        encoded_origins[origin] = {}
        for key, site in sites.items():
            encoded_origins[origin][key] = {**{k: v for k, v in site.items() if k != "tensors"},
                "tensors": {name: _encode_tensor(value) for name, value in site["tensors"].items()}}
    write_json(path, {"schema": "olmo-boundary-precision-fixture-v1", "provenance": provenance,
                      "origins": encoded_origins})
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            "tensor_payloads": sum(len(site["tensors"]) for sites in origins.values() for site in sites.values()),
            "sites_per_origin": len(next(iter(origins.values())))}


def local_vjp(model, site, common, *, precision):
    """Original module computation on independent boundary leaves and fixed dY.

This keeps master parameters attached but uses autograd.grad, not .backward:
model .grad must remain None. Unused stack embedding/readout weights are named
explicitly and omitted as mathematical zeros from the returned used gradients.
"""
    if precision not in ("fp32", "bf16_mixed") or site["kind"] != common["kind"]:
        raise ValueError("Unsupported local boundary comparison")
    module = model.backbone.fusion if site["kind"] == "fusion" else model.backbone.backbone
    parameters = dict(module.named_parameters())
    if any(p.dtype != torch.float32 or not p.requires_grad or p.grad is not None for p in parameters.values()):
        raise ValueError("Local VJP needs attached FP32 trainable masters with .grad=None")
    device = next(module.parameters()).device
    names = site["input_names"]
    if any(site["tensors"][name].dtype != torch.float32 for name in names) or common["tensors"]["cotangent"].dtype != torch.float32:
        raise ValueError("Boundary leaves and common cotangent must actually be FP32")
    inputs = {name: fixed_leaf(site["tensors"][name], site["layouts"][name], device=device,
                               dtype=torch.float32, requires_grad=True) for name in names}
    cotangent = fixed_leaf(common["tensors"]["cotangent"], common["layouts"]["cotangent"],
                           device=device, dtype=torch.float32, requires_grad=False)
    tensor_kwargs = {name: fixed_leaf(site["tensors"][name], site["layouts"][name], device=device,
                        dtype=site["tensors"][name].dtype, requires_grad=False) for name in site["tensor_kwarg_names"]}
    pins = tree_digests({**inputs, **tensor_kwargs, "cotangent": cotangent})
    values_match = all(pins[name] == tree_digests(site["tensors"][name]) for name in (*names, *tensor_kwargs))
    values_match &= pins["cotangent"] == tree_digests(common["tensors"]["cotangent"])
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16_mixed", cache_enabled=False):
        if site["kind"] == "fusion":
            output = module(**inputs)
        else:
            kwargs = {**site["kwargs"], **tensor_kwargs, **inputs}
            kwargs["mode"] = RTMode(**kwargs["mode"])
            output = module(**kwargs).last_hidden_state
    gradients = torch.autograd.grad(output, (*inputs.values(), *parameters.values()), cotangent, allow_unused=True)
    if any(value is None for value in gradients[:len(inputs)]):
        raise AssertionError("A boundary input unexpectedly has no VJP")
    prefix = "backbone.fusion." if site["kind"] == "fusion" else "backbone.backbone."
    used, unused = {}, []
    for (name, parameter), value in zip(parameters.items(), gradients[len(inputs):]):
        if value is None:
            unused.append(prefix+name)
        else:
            used[prefix+name] = value.detach().cpu().clone()
    expected_unused = [] if site["kind"] == "fusion" else [prefix+name for name, p in parameters.items() if p is module.readout_weight]
    values = {"output": output.detach().cpu().clone(), "inputs": {
        name: value.detach().cpu().clone() for name, value in zip(names, gradients[:len(inputs)])}, "parameters": used}
    health = {"input_and_cotangent_values_exact": values_match,
              "local_inputs_and_cotangent_unchanged": pins == tree_digests({**inputs, **tensor_kwargs, "cotangent": cotangent}),
              "expected_unused_parameters": unused == expected_unused,
              "parameter_grads_remain_none": all(p.grad is None for p in model.parameters()),
              "finite": all(bool(torch.isfinite(value).all()) for value in (values["output"], *values["inputs"].values(), *used.values()))}
    return {"values": values, "health": health, "input_cotangent_pins": pins,
            "input_layouts": {name: layout_record(value) for name, value in inputs.items()},
            "leaf_scope": "Independent leaves preserve values/shapes/strides/dtypes; original storage offsets and aliasing are not preserved",
            "unused_parameters": unused, "used_parameter_tensors": len(used),
            "parameter_vjp_scope": "All used module parameters; listed unused parameters are zero and omitted from vectors"}


def compare_vjps(actual, reference, valid):
    return {"output_all": tensor_geometry(actual["output"], reference["output"]),
            "output_valid": tensor_geometry(actual["output"][valid], reference["output"][valid]),
            "input_vjps": {name: tensor_geometry(value, reference["inputs"][name]) for name, value in actual["inputs"].items()},
            "parameter_vjps": gradient_geometry(actual["parameters"], reference["parameters"])}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.reference_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.reference_sha256):
        parser.error("Reference SHA256 must contain 64 lowercase hexadecimal characters")
    return args


def source_hashes():
    sources = {**fusion_sources(), **attention_sources()}
    for name in ("scripts/olmo_campaign_boundary_precision.py", "tests/test_campaign_boundary_precision.py",
                 "docs/reports/olmo-boundary-precision/protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def main(argv=None):
    args = parse_args(argv)
    deterministic = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Boundary diagnostic is single-process, without DDP")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "olmo-campaign-boundary-precision-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "scope": __doc__, "sources": source_hashes(),
        "runtime": runtime, "determinism": deterministic, "reference_sha256": args.reference_sha256,
        "rows": [], "aggregate_anchor_backwards": 2, "physical_anchor_backwards": 4, "local_vjps": 12,
        "optimizer_updates": 0, "paths": PATHS, "record": 0, "passes": PASSES,
        "qualification": "Descriptive module-level comparisons, not backward-kernel fault attribution or numerical clearance; errors are not additive",
        "geometry_scope": "Full boundary tensors/VJPs use actual masks and unchanged common BF16-origin cotangent; output_valid is a summary selection only",
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for name in report["sources"]:
        target = args.output_dir/"source-snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-boundary-precision", name=args.output_dir.name, preserve_state=preserve_local_rng)
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
            raise AssertionError("Boundary operational/reference guard failed")

    try:
        tracker.start({key: report[key] for key in ("scope", "paths", "reference_sha256", "determinism", "qualification")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("verify_reference_and_load_source")
        previous = load_reference(args.reference_report, args.reference_sha256, report["sources"])
        model, recipe, checkpoint, ids, eos = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NF", torch.device("cuda"))
        original = {key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS}
        fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
                    length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
        info = {"contract": arm_contract(model, recipe), "source_checkpoint": checkpoint,
            "model_config": model.backbone.backbone.config.to_dict(), "nextlat_config": model.config.to_dict(),
            "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256, "initial_state": state_pins(model),
            "production_runtime_flags": original, "fixture_inputs": fixture_pins(fixtures),
            "fixture_metadata": global_fixture_metadata(model, fixtures)}
        contracts = state_contract_checks(info, previous["arms"]["NF"])
        contracts.update(runtime=runtime == previous["runtime"], determinism=deterministic == previous["determinism"],
            math_reduction=report["math_sdpa_reduced_precision_reduction"] == previous["math_sdpa_reduced_precision_reduction"],
            bf16_matmul_reduction=report["bf16_matmul_reduced_precision_reduction"] == previous["bf16_matmul_reduced_precision_reduction"])
        report.update(arm_state=info, reference_contract_checks=contracts)
        if not all(contracts.values()):
            raise AssertionError("NF anchor state/runtime/input contract differs")
        rng, origins = rng_snapshot(), {}
        training_modes = {name: module.training for name, module in model.named_modules()}
        report["training_modes"] = training_modes
        reference_gradients, reference_forward = None, None
        for path in (FP32, BF16):
            execution = configure_path(model, original, path)
            persist("anchor/"+path)
            backend = SDPBackend.MATH if path == FP32 else SDPBackend.FLASH_ATTENTION
            with capture_boundaries(model, fixtures) as boundaries, capture_passes(model, fixtures) as passes, sdpa_kernel(backend):
                metrics = component_backward(model, recipe, fixtures, precision=execution["precision"], layout="sparse", objective="ce")
            gradients, snapshot = record_gradients(model, reference_gradients, save_cpu=path == FP32,
                scope="Exact NF anchor reconstruction; no new budget")
            forward = forward_geometry(passes, reference_forward)
            fingerprints = tree_digests([{k: record[k] for k in ("batch", "token_embeddings", "pass_hidden_states")} for record in passes])
            comparison = anchor_comparison(metrics, fingerprints, gradients, forward, reference_anchor(previous, path))
            origins[path] = boundaries["sites"]
            fixed_inputs = fixture_pins(fixtures) == info["fixture_inputs"]
            publish("anchor/"+path, {"execution": execution, "metrics": metrics, "gradients": gradients,
                "anchor_comparison": comparison, "observer": {k: v for k, v in boundaries.items() if k != "sites"},
                "inputs_unchanged": fixed_inputs, "rng_unchanged": rng_unchanged(rng),
                "passed": comparison["passed"] and gradients["finite"] and fixed_inputs and rng_unchanged(rng)})
            if path == FP32:
                reference_gradients, reference_forward = snapshot, passes
        model.zero_grad(set_to_none=True)
        del reference_gradients, reference_forward, snapshot, passes
        gc.collect()
        match_sites(origins[FP32], origins[BF16])
        captured_pins = tree_digests(origins)
        fixture_path = args.output_dir/"boundary-fixture.json"
        report["boundary_fixture"] = write_fixture(fixture_path, origins,
            provenance={"reference_sha256": args.reference_sha256, "sources": report["sources"],
                        "source_checkpoint": checkpoint, "recipe_sha256": recipe.sha256})
        versions = [(p.data_ptr(), p._version, p.dtype) for p in (*model.parameters(), *model.buffers())]
        for key in origins[BF16]:
            references = {}
            a, b = origins[FP32][key], origins[BF16][key]
            inherited_inputs = {name: tensor_geometry(b["tensors"][name], a["tensors"][name]) for name in a["input_names"]}
            for index, path in enumerate(PATHS):
                source = a if index == 0 else b
                execution = configure_path(model, original, BF16 if index == 2 else FP32)
                persist(key+"/"+path)
                backend = SDPBackend.FLASH_ATTENTION if index == 2 else SDPBackend.MATH
                with sdpa_kernel(backend):
                    result = local_vjp(model, source, b, precision=execution["precision"])
                values = result.pop("values")
                health = result["health"]
                health["rng_unchanged"] = rng_unchanged(rng)
                health["parameter_versions_unchanged"] = versions == [(p.data_ptr(), p._version, p.dtype) for p in (*model.parameters(), *model.buffers())]
                health["training_modes_unchanged"] = training_modes == {name: module.training for name, module in model.named_modules()}
                if index in (0, 2):
                    health["origin_output_bitwise_exact"] = tree_digests(values["output"]) == tree_digests(source["tensors"]["output"])
                row = {"site": key, "path": path, "execution": execution, "local": result,
                    "captured_layouts": source["layouts"], "inherited_input_perturbation": inherited_inputs,
                    "value_pins": tree_digests(values), "output": tensor_geometry(values["output"]),
                    "input_vjps": {name: tensor_geometry(value) for name, value in values["inputs"].items()},
                    "health": health, "memory": memory(), "passed": all(health.values())}
                for previous_path, reference in references.items():
                    row["versus_"+previous_path] = compare_vjps(values, reference, b["tensors"]["valid_output_mask"])
                references[path] = values
                publish(key+"/"+path, row)
            del references, values, reference
            gc.collect()
        health = {"parameters_and_buffers_unchanged": state_pins(model) == info["initial_state"],
            "original_inputs_unchanged": fixture_pins(fixtures) == info["fixture_inputs"],
            "captured_boundaries_unchanged": captured_pins == tree_digests(origins),
            "parameter_grads_remain_none": all(p.grad is None for p in model.parameters()),
            "training_modes_unchanged": training_modes == {name: module.training for name, module in model.named_modules()},
            "production_flags_restored": original == {key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS},
            "sources_unchanged": source_hashes() == report["sources"], "rng_unchanged": rng_unchanged(rng),
            "reference_unchanged": sha256_file(args.reference_report) == args.reference_sha256,
            "fixture_unchanged": sha256_file(fixture_path) == report["boundary_fixture"]["sha256"]}
        publish("final_integrity", {**health, "passed": all(health.values())})
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
