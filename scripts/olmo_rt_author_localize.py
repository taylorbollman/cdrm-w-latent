#!/usr/bin/env python3
"""Localize author/native RT arithmetic on fixed actual pretrained block-0 x/g."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained import olmo_author, olmo_tiled
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo_rope import build_rope_tables
from cdrm.pretrained.static_training import normalized_objective
from scripts.olmo_lm_common import tree_digests, tensor_digest
from scripts.olmo_f3d_validate import global_gradient_l2, new_plan
from scripts.olmo_f4_resources import selected_case
from scripts.olmo_rt_author_compare import (configure_compiler, compiler_audit, tensor_metrics,
    FP32_BUDGETS, BF16_BUDGETS, fp32_tensor_passes)
from scripts.olmo_rt_author_integration import (SOURCES as INTEGRATION_SOURCES, SEED, full_batch,
    set_backend, parameter_signature, build_model, configure_determinism, require_container_gpu,
    validate_prepared_manifest, load_native_state_dict, load_native_tokenizer, backend_context, OnlineTracker)

PROTOCOL = ROOT / "docs/reports/olmo-rt-author-integration/localization-protocol.md"
SOURCES = tuple(sorted(set(INTEGRATION_SOURCES) | {"scripts/olmo_rt_author_localize.py"}))
ARMS = ("native_fp32", "author_fp32", "native_mixed", "author_legacy_mixed",
        "author_fp32_state_mixed", "author_legacy_separate_self_mixed")


def normalize_gaussian(reference, *, seed=SEED + 91):
    """Keep the incoming adjoint's global L2 scale; change only its direction."""
    if reference.dtype != torch.float32 or not bool(torch.isfinite(reference).all()):
        raise ValueError("Reference cotangent must be finite FP32")
    norm = reference.double().norm()
    if not bool(norm > 0):
        raise ValueError("Reference cotangent must have positive L2 norm")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(reference.shape, generator=generator, dtype=torch.float32).to(reference.device)
    return value * (norm / value.double().norm()).to(torch.float32)


def _separate_self_atts(final_v, v_init, alphas, fp32_state):
    """Diagnostic only: remove permanent self before PV, add temporary self.

    Keep input/output dtypes and the original helper's fullgraph boundary.
    A changed graph can induce changed compiler fusion/rounding; this is not an
    instruction-level attribution experiment or a proposed production policy.
    """
    if fp32_state:
        raise ValueError("Separate-self diagnostic is bounded to author_legacy")
    historical = alphas.clone()
    historical.diagonal(dim1=2, dim2=3).zero_()
    attention = (final_v.permute(1, 2, 3, 0) @ historical).permute(3, 0, 1, 2)
    self_probability = alphas.diagonal(dim1=2, dim2=3).permute(2, 0, 1).unsqueeze(-1)
    return attention + v_init * self_probability


_COMPILED_SEPARATE_SELF = torch.compile(_separate_self_atts, fullgraph=True)


@contextmanager
def _observe_reconstruction(*, separate_self):
    """Invocation-local helper override, restored even after failure."""
    observed = {}
    original = olmo_author._helper
    def helper(spec, name, *args):
        if separate_self and name == "recompute_atts":
            result = (_COMPILED_SEPARATE_SELF if spec.compiled_helpers else _separate_self_atts)(*args)
        else:
            result = original(spec, name, *args)
        if name == "recompute_atts":
            # At position zero there is no history: exact attention is v_init.
            observed["token0_vs_temporary_value"] = tensor_metrics(result[0].detach().cpu(), args[1][0].detach().cpu())
        return result
    with patch.object(olmo_author, "_helper", helper):
        yield observed


def local_vjp(layer, x, g, tables, arm, *, compiled_helpers=True):
    """Same fixed x/g, explicit autograd.grad, no original .grad side effects.

    CPU calls are explicit test fixtures (compiled_helpers=False); main never
    falls back from CUDA. Provided tables are authoritative actual positions.
    Integer coordinates below are metadata-only because native RoPE reuse is on.
    """
    if arm not in ARMS:
        raise ValueError("Unknown local diagnostic arm")
    if x.shape != g.shape or x.dtype != torch.float32 or g.dtype != torch.float32 or x.device != g.device:
        raise ValueError("Fixed input/cotangent must be same-shaped/device FP32 tensors")
    if x.device.type == "cpu" and compiled_helpers:
        raise ValueError("CPU fixtures must explicitly disable compiled helpers")
    named = tuple(layer.named_parameters())
    trainable = tuple((name, weight) for name, weight in named if weight.requires_grad)
    ownership = tuple((name, id(weight), weight.data_ptr(), weight._version, weight.requires_grad,
                       None if weight.grad is None else (id(weight.grad), weight.grad._version)) for name, weight in named)
    x_local = x.detach().clone().requires_grad_(True)
    positions = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
    valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    mixed = arm.endswith("mixed")
    separate_self = arm == "author_legacy_separate_self_mixed"
    with _observe_reconstruction(separate_self=separate_self) as reconstruction:
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=mixed, cache_enabled=False):
            if arm.startswith("native"):
                tiles = "triton" if mixed and x.device.type == "cuda" else "eager"
                output, _ = olmo_tiled.tiled_recurrent_layer(layer, x_local, alpha=1, past=None,
                    query_positions=positions, key_positions=positions, key_valid=valid,
                    attention_precision="mixed" if mixed else "fp32", cast_weights_once=True,
                    tile_backend=tiles, backward_tile_backend=tiles, backward_memory="recompute",
                    reuse_rope=True, kv_only_writes=True, query_rope=tables, key_rope=tables)
            else:
                output = olmo_author.author_tiled_recurrent_layer(layer, x_local, tables,
                    compiled_helpers=compiled_helpers, bwd_mlp_chunks=4,
                    precision_policy="fp32_state" if arm == "author_fp32_state_mixed" else "author_legacy",
                    autocast_cache=True)
        gradients = torch.autograd.grad(output, (x_local, *(weight for _, weight in trainable)), g)
    after = tuple((name, id(weight), weight.data_ptr(), weight._version, weight.requires_grad,
                   None if weight.grad is None else (id(weight.grad), weight.grad._version)) for name, weight in named)
    if ownership != after:
        raise AssertionError("Local VJP changed original parameter/gradient ownership or values")
    result = {"arm": arm, "output": output.detach().cpu(), "input_gradient": gradients[0].detach().cpu(),
        "parameter_gradients": {name: gradient.detach().cpu() for (name, _), gradient in zip(trainable, gradients[1:])},
        "ownership_preserved": True, "reconstruction": reconstruction}
    result["finite"] = all(bool(torch.isfinite(tensor).all()) for tensor in
        (result["output"], result["input_gradient"], *result["parameter_gradients"].values()))
    return result


def compare_local(candidate, reference, *, name, policy="diagnostic"):
    if policy not in {"diagnostic", "fp32", "bf16"}:
        raise ValueError("Unknown comparison policy")
    names = set(reference["parameter_gradients"])
    ownership = names == set(candidate["parameter_gradients"])
    gradients = {key: tensor_metrics(candidate["parameter_gradients"][key], reference["parameter_gradients"][key])
                 for key in sorted(names & candidate["parameter_gradients"].keys())}
    output = tensor_metrics(candidate["output"], reference["output"])
    input_gradient = tensor_metrics(candidate["input_gradient"], reference["input_gradient"])
    rows = [output, input_gradient, *gradients.values()]
    finite = all(row["finite"] for row in rows)
    global_l2 = global_gradient_l2(gradients.values())
    passed = ownership and finite and bool(gradients)
    if policy == "fp32":
        passed = passed and all(fp32_tensor_passes(row) for row in rows) and (
            global_l2 <= FP32_BUDGETS["relative_l2"] or math.sqrt(sum(row["delta_sq"] for row in gradients.values())) <= FP32_BUDGETS["absolute_l2"])
    elif policy == "bf16":
        passed = (passed and global_l2 <= BF16_BUDGETS["global_parameter_relative_l2"]
            and all(row["relative_l2"] <= BF16_BUDGETS["gradient_relative_l2"]
                and row["max_relative"] <= BF16_BUDGETS["gradient_max_relative"] for row in [input_gradient, *gradients.values()])
            and output["relative_l2"] <= BF16_BUDGETS["output_relative_l2"]
            and output["max_relative"] <= BF16_BUDGETS["output_max_relative"])
    return {"name": name, "policy": policy, "gate": False, "passed": passed, "finite": finite,
        "ownership_matches": ownership, "global_parameter_relative_l2": global_l2,
        "output": output, "input_gradient": input_gradient, "gradients": gradients,
        "candidate": candidate["arm"], "reference": reference["arm"],
        "budgets": FP32_BUDGETS if policy == "fp32" else BF16_BUDGETS if policy == "bf16" else None,
        "qualification": "Fixed external cotangent; no MSE/CE objective comparison applies. A diagnostic pass means finite/owned, not numerical equivalence."}


def statistics(value):
    cpu = value.detach().float().cpu()
    return {**tree_digests(cpu), "finite": bool(torch.isfinite(cpu).all()),
        "l2": float(cpu.double().norm()), "rms": float(cpu.double().square().mean().sqrt()),
        "maximum_abs": float(cpu.abs().max()), "minimum": float(cpu.min()), "maximum": float(cpu.max())}


def compare_parameter_gradients(candidate, reference):
    """Report exact replay and raw metrics, without treating norms as equality."""
    ownership = bool(reference) and set(candidate) == set(reference)
    gradients = {name: tensor_metrics(candidate[name], reference[name])
                 for name in sorted(candidate.keys() & reference.keys())}
    finite = bool(gradients) and all(row["finite"] for row in gradients.values())
    exact = ownership and finite and all(row["bitwise_equal"] for row in gradients.values())
    return {"ownership_matches": ownership, "finite": finite, "all_bitwise_equal": exact,
        "global_parameter_relative_l2": global_gradient_l2(gradients.values()), "gradients": gradients,
        "qualification": "Exact replay is established only when all_bitwise_equal is true; matching norms alone is insufficient."}


def capture_native_fixture(plan, *, backend="native"):
    """Capture one native/author pass; default preserves the original API."""
    if backend not in {"native", "author"}:
        raise ValueError("Capture backend must be native or author")
    target = plan.model.backbone.backbone.layers[0]
    module = olmo_tiled if backend == "native" else olmo_author
    function = "tiled_recurrent_layer" if backend == "native" else "author_tiled_recurrent_layer"
    original = getattr(module, function)
    observed = {}
    def layer_call(layer, x, *args, **kwargs):
        result = original(layer, x, *args, **kwargs)
        output = result[0] if backend == "native" else result
        if layer is target:
            if observed:
                raise AssertionError("Expected one block-0 invocation in one RT-only pass")
            observed.update(x=x.detach().clone(), output=output.detach().clone())
            def save_cotangent(gradient):
                observed["cotangent"] = gradient.detach().clone()
            output.register_hook(save_cotangent)
        return result
    plan.validate_execution()
    with patch.object(module, function, layer_call):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            result = plan.loss_sums()
            objective = normalized_objective(result)
        objective.backward()
    if set(observed) != {"x", "output", "cotangent"}:
        raise AssertionError("Did not capture exact block-0 input/output/incoming cotangent")
    return observed, {"objective": float(objective.detach()), "ce_sum": float(result.sums["ce"].detach()),
        "counts": dict(plan.counts), "input_tokens": plan.input_tokens,
        "backend": backend,
        "ordinary_flash": "deterministic Flash context, same frozen integration stack",
        "backward_calls": 1, "physical_optimizer_updates": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    began = time.perf_counter()
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    compilation = configure_compiler()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for source in SOURCES:
        destination = args.output_dir / "source-snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    shutil.copyfile(PROTOCOL, args.output_dir / "protocol.md")
    report = {"schema": "olmo-rt-author-localization-v1", "status": "running", "stage": "load",
        "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": runtime, "determinism": determinism, "compiler_configuration": compilation,
        "configuration": {"batch_size": 8, "length": 512, "case": "rt", "selected_rt_layers": [0, 15],
            "supervision": "full", "ce_chunk_size": 2048, "seed": SEED, "gaussian_seed": SEED + 91,
            "fixed_block": 0, "normalization": "actual mean-CE cotangent; Gaussian matched to its global L2",
            "arms": list(ARMS), "tf32": False, "physical_optimizer_updates": 0,
            "full_model_backward_calls": 2, "local_vjp_calls": 9,
            "additional_author_own_cotangent": True},
        "source_hashes": {source: sha256_file(ROOT / source) for source in SOURCES},
        "protocol_sha256": sha256_file(PROTOCOL), "started_utc": datetime.now(timezone.utc).isoformat(),
        "physical_optimizer_updates": 0, "checks": [], "local_snapshots": []}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-rt-author-integration",
        name=args.output_dir.name, output_dir=args.output_dir)
    def save(stage=None):
        if stage is not None:
            report["stage"] = stage
            print({"stage": stage, "elapsed_seconds": time.perf_counter() - began}, flush=True)
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)
    def publish(check):
        report["checks"].append(check)
        save()
        tracker.log({"localization/global_gradient_relative_l2": check["global_parameter_relative_l2"],
            "localization/input_gradient_relative_l2": check["input_gradient"]["relative_l2"],
            "localization/output_relative_l2": check["output"]["relative_l2"]}, step=len(report["checks"]))
        print({"comparison": check["name"], "global_gradient_relative_l2": check["global_parameter_relative_l2"],
               "input_gradient_relative_l2": check["input_gradient"]["relative_l2"], "screen_passed": check["passed"]}, flush=True)
    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": report["configuration"], "checkpoint": report["checkpoint"]})
        save()
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        case = selected_case("rt", batch=8, length=512)
        model = build_model(state, case)
        del state
        set_backend(model, "native")
        batch = full_batch(tokenizer, case, 0)
        report["batch"] = tree_digests(vars(batch))
        before = parameter_signature(model)
        versions = tuple((name, parameter._version) for name, parameter in model.named_parameters())
        with backend_context("flash"):
            plan = new_plan(model, batch, case.mode(), "recompute")
            report["prepared_layout"] = plan.forward_layout.metadata
            save("capture_actual_native_input_and_cotangent")
            captured, capture = capture_native_fixture(plan)
        report["capture"] = capture
        if parameter_signature(model) != before:
            raise AssertionError("Capturing the native fixture changed parameter ownership")
        x, g = captured["x"], captured["cotangent"]
        positions = plan.forward_layout.position_ids.detach().clone()
        tables = build_rope_tables(positions, model.backbone.config.head_dim, model.backbone.config.rope_freq_constant)
        layer = model.backbone.backbone.layers[0]
        layer_hashes = tree_digests(dict(layer.named_parameters()))
        captured_output = captured["output"].detach().cpu()
        native_full_gradients = {name: parameter.grad.detach().cpu().clone()
            for name, parameter in layer.named_parameters() if parameter.grad is not None}
        model.zero_grad(set_to_none=True)
        del plan, captured
        set_backend(model, "author")
        with backend_context("flash"):
            plan = new_plan(model, batch, case.mode(), "recompute")
            report["author_prepared_layout"] = plan.forward_layout.metadata
            save("capture_actual_author_input_and_cotangent")
            author_captured, author_capture = capture_native_fixture(plan, backend="author")
        report["author_capture"] = author_capture
        author_g = author_captured["cotangent"]
        author_captured_output = author_captured["output"].detach().cpu()
        report["capture_inputs_identical"] = torch.equal(author_captured["x"], x)
        report["capture_positions_identical"] = torch.equal(plan.forward_layout.position_ids, positions)
        report["full_model_parameters_unchanged"] = (parameter_signature(model) == before and versions ==
            tuple((name, parameter._version) for name, parameter in model.named_parameters()))
        if not all(report[key] for key in ("capture_inputs_identical", "capture_positions_identical", "full_model_parameters_unchanged")):
            raise AssertionError("Native/author captures changed input/positions/weight state")
        report["actual_cotangent_comparison"] = tensor_metrics(author_g.detach().cpu(), g.detach().cpu())
        report["captured_block_output_comparison"] = tensor_metrics(author_captured_output, captured_output)
        author_full_gradients = {name: parameter.grad.detach().cpu().clone()
            for name, parameter in layer.named_parameters() if parameter.grad is not None}
        report["full_model_block0_parameter_comparison"] = compare_parameter_gradients(author_full_gradients, native_full_gradients)
        model.zero_grad(set_to_none=True)
        del plan, model, author_captured, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        random_g = normalize_gaussian(g)
        report["fixed_tensors"] = {"input": statistics(x), "actual_cotangent": statistics(g),
            "author_actual_cotangent": statistics(author_g),
            "gaussian_cotangent": statistics(random_g), "positions": tree_digests(positions)}
        report["block_parameters"] = layer_hashes
        fixed_hashes = (tensor_digest(x), tensor_digest(g), tensor_digest(random_g), tensor_digest(author_g))
        snapshots = {}
        for arm in ARMS:
            save("actual_cotangent/" + arm)
            start = time.perf_counter()
            snapshot = local_vjp(layer, x, g, tables, arm)
            if not snapshot["finite"]:
                raise FloatingPointError("Nonfinite local diagnostic: " + arm)
            report["local_snapshots"].append({"direction": "actual", "arm": arm,
                "seconds": time.perf_counter() - start, "finite": snapshot["finite"],
                "ownership_preserved": snapshot["ownership_preserved"], "reconstruction": snapshot["reconstruction"]})
            if arm != "native_fp32":
                publish(compare_local(snapshot, snapshots["native_fp32"], name=arm + "_vs_native_fp32",
                    policy="fp32" if arm == "author_fp32" else "diagnostic"))
            if arm.startswith("author") and arm.endswith("mixed"):
                publish(compare_local(snapshot, snapshots["native_mixed"], name=arm + "_vs_native_mixed", policy="bf16"))
            if arm == "native_mixed":
                report["native_local_matches_captured_forward"] = tensor_metrics(snapshot["output"], captured_output)
                if not report["native_local_matches_captured_forward"]["bitwise_equal"]:
                    raise AssertionError("Local native forward differs from captured block-0 forward")
                report["native_local_vs_full_model_parameter_gradients"] = compare_parameter_gradients(
                    snapshot["parameter_gradients"], native_full_gradients)
            if arm == "author_legacy_separate_self_mixed":
                check = tensor_metrics(snapshot["output"], snapshots["author_legacy_mixed"]["output"])
                report["separate_self_forward_unchanged"] = check
                if not check["bitwise_equal"]:
                    raise AssertionError("Backward-only diagnostic changed forward")
                publish(compare_local(snapshot, snapshots["author_legacy_mixed"], name="separate_self_vs_legacy_mixed"))
            if arm in {"native_fp32", "native_mixed", "author_legacy_mixed"}:
                snapshots[arm] = snapshot
            save()
        save("author_own_cotangent/author_legacy_mixed")
        own_author = local_vjp(layer, x, author_g, tables, "author_legacy_mixed")
        if not own_author["finite"]:
            raise FloatingPointError("Nonfinite author own-cotangent local diagnostic")
        report["local_snapshots"].append({"direction": "author_actual", "arm": "author_legacy_mixed",
            "finite": own_author["finite"], "ownership_preserved": own_author["ownership_preserved"],
            "reconstruction": own_author["reconstruction"]})
        report["author_local_matches_captured_forward"] = tensor_metrics(own_author["output"], author_captured_output)
        report["author_local_vs_full_model_parameter_gradients"] = compare_parameter_gradients(
            own_author["parameter_gradients"], author_full_gradients)
        own_author["arm"] = "author_legacy_mixed_own_cotangent"
        publish(compare_local(own_author, snapshots["author_legacy_mixed"], name="author_own_vs_native_cotangent"))
        report["own_cotangent_replays_exact"] = all(report[key]["all_bitwise_equal"] for key in (
            "native_local_vs_full_model_parameter_gradients", "author_local_vs_full_model_parameter_gradients")) and report["author_local_matches_captured_forward"]["bitwise_equal"]
        report["own_cotangent_replay_qualification"] = "Incoming-trajectory attribution requires exact own-cotangent parameter replay; any mismatch remains unresolved."
        del own_author, native_full_gradients, author_full_gradients
        del snapshots, snapshot
        for arm in ("native_mixed", "author_legacy_mixed"):
            save("gaussian_cotangent/" + arm)
            snapshot = local_vjp(layer, x, random_g, tables, arm)
            report["local_snapshots"].append({"direction": "normalized_gaussian", "arm": arm,
                "finite": snapshot["finite"], "ownership_preserved": snapshot["ownership_preserved"],
                "reconstruction": snapshot["reconstruction"]})
            if not snapshot["finite"]:
                raise FloatingPointError("Nonfinite Gaussian-cotangent local diagnostic")
            if arm == "native_mixed":
                gaussian_reference = snapshot
            else:
                publish(compare_local(snapshot, gaussian_reference, name="gaussian_direction_author_vs_native_mixed", policy="bf16"))
        report["fixed_tensors_unchanged"] = fixed_hashes == (tensor_digest(x), tensor_digest(g), tensor_digest(random_g), tensor_digest(author_g))
        report["block_parameters_unchanged"] = layer_hashes == tree_digests(dict(layer.named_parameters()))
        if not report["fixed_tensors_unchanged"] or not report["block_parameters_unchanged"]:
            raise AssertionError("Diagnostic changed its fixed input/cotangent/weight fixture")
        report["compiler_final"] = compiler_audit(required=True)
        if report["source_hashes"] != {source: sha256_file(ROOT / source) for source in SOURCES} or report["protocol_sha256"] != sha256_file(PROTOCOL):
            raise AssertionError("Frozen localization source/protocol changed")
        report["status"] = "completed"
        report["qualification"] = "Completed diagnostic, not numerical clearance. Failed existing screens remain failed; fixed g removes downstream feedback differences but each local arm retains its own recurrent forward trajectory."
        tracker.summary({"result_status": "completed", "physical_optimizer_updates": 0})
        save("complete")
    except BaseException as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        save()
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "completed")
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            report["elapsed_seconds"] = time.perf_counter() - began
            save()


if __name__ == "__main__":
    main()
