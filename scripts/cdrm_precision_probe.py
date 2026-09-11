#!/usr/bin/env python3
"""Fixed-state CDRM precision controls; diagnostics only, never training.

Every arm starts from the same archived checkpoint and numerical minibatch.
Ordinary-call precision wrappers preserve the shared fabric parameter owners.
The original numerical screens are reported without changing their thresholds.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import copy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm_tiled_common import (build_model, loss_sum, optimizer_for, setup,
                               snapshot_sources, sources, verify_sources)
from cdrm_tiled_validate import (adam_comparison, comparison, cpu, digest,
                                 full_packet, gradients, metric, preserve_rng,
                                 save_json)
from experiment_tracking import OnlineTracker
from r3_backward_validate import adam_state, step_packet
from stage_a_common import configure_compiled_helpers
from stage_b_train import compiler_audit


ADAPTERS = ("cdrm.deep_adapter.weight", "cdrm.bridge_adapter.weight")
BASELINES = ("current_fp32", "current_bf16")
CONTRACT = Path("docs/reports/cdrm-numerical-resolution/reference-contract.md")
CONTRACT_SHA256 = "6660b964a2f5a678d4324ac2799c814ad617ce9acceabbed5e0b0e23101714cc"


@dataclass(frozen=True)
class Arm:
    name: str
    bf16: bool
    lambda_zero: bool = False
    bypass: bool = False
    reduction_off: bool = False
    fp32_blocks: tuple[int, ...] = ()
    fp32_head: bool = False
    fp32_bias: bool = False
    fp32_attention_blocks: tuple[int, ...] = ()


ARMS = {arm.name: arm for arm in (
    Arm("current_fp32", False), Arm("current_bf16", True),
    Arm("fp32_lambda_zero", False, lambda_zero=True),
    Arm("bf16_lambda_zero", True, lambda_zero=True),
    Arm("fp32_bypass", False, bypass=True), Arm("bf16_bypass", True, bypass=True),
    Arm("bf16_reduction_off", True, reduction_off=True),
    Arm("bf16_side_fp32_backbone", True, fp32_blocks=(0, 1, 2, 3, 4), fp32_head=True),
    *(Arm(f"bf16_block_{index}_fp32", True, fp32_blocks=(index,)) for index in range(5)),
    Arm("bf16_alibi_fp32", True, fp32_bias=True),
    Arm("bf16_attention_fp32", True, fp32_attention_blocks=(0, 1, 2, 3, 4)),
)}
COARSE = ("current_fp32", "current_bf16", "fp32_lambda_zero", "bf16_lambda_zero",
          "fp32_bypass", "bf16_bypass", "bf16_reduction_off", "bf16_side_fp32_backbone")


def selected_arms(requested):
    selected = list(BASELINES)
    for name in requested or COARSE:
        if name not in ARMS:
            raise ValueError(f"Unknown fixed control: {name}")
        if ARMS[name].lambda_zero or ARMS[name].bypass:
            reference = "fp32_bypass" if ARMS[name].bypass else "fp32_lambda_zero"
            if reference not in selected:
                selected.append(reference)
        if name not in selected:
            selected.append(name)
    # A bypass must also be compared with an explicitly zeroed fabric.
    for name in tuple(selected):
        if ARMS[name].bypass:
            reference = "bf16_lambda_zero" if ARMS[name].bf16 else "fp32_lambda_zero"
            if reference not in selected:
                selected.append(reference)
    return tuple(selected)


@contextmanager
def replace_method(module, name, replacement):
    """Restore the original instance/class lookup, including after an exception."""
    own = name in module.__dict__
    previous = module.__dict__.get(name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        if own:
            setattr(module, name, previous)
        else:
            delattr(module, name)


def remember_tensor(record, name, tensor):
    record[name] = cpu(tensor)
    record.setdefault("dtypes", {})[name] = str(tensor.dtype)
    if tensor.requires_grad:
        def receive(gradient):
            key = f"{name}_gradient"
            record[key] = cpu(gradient)
            calls = record.setdefault("gradient_hook_calls", {})
            calls[name] = calls.get(name, 0) + 1
            # Returning None preserves the native adjoint.
        tensor.register_hook(receive)


def ordinary_attention_fp32(block, x, attention_bias=None, layer_past=None,
                            use_cache=False, max_doc_len=None, cu_doc_lens=None):
    """The existing pre-norm ordinary block, with its attention region in FP32.

This runtime control reproduces _real_forward's operations for the frozen
profile. It never replaces shared child modules and leaves the MLP under the
caller's original autocast context. It is not a production precision policy.
"""
    cfg = block.config
    if (cfg.norm_after or block._activation_checkpoint_fn is not None
            or cfg.attention_dropout or cfg.residual_dropout or cfg.clip_qkv is not None
            or layer_past is not None or use_cache or max_doc_len is not None or cu_doc_lens is not None):
        raise ValueError("Attention-region control requires the frozen unpadded pre-norm, dropout-free profile")
    with torch.autocast(x.device.type, enabled=False):
        h = block.attn_norm(x.float())
        qkv = block.att_proj(h)
        q, k, v = qkv.split(block.fused_dims, dim=-1)
        attention, cache = block.attention(q, k, v, attention_bias, layer_past=None,
                                           use_cache=False, max_doc_len=None, cu_doc_lens=None)
        x = x + block.dropout(attention)
    h = block.ff_norm(x)
    h = block.ff_proj(h)
    h = block.act(h)
    h = block.ff_out(h)
    return x + block.dropout(h), cache


@contextmanager
def ordinary_controls(model, arm, capture=None):
    """Wrap ordinary block calls, never the owner's shared child modules."""
    canonical = tuple((name, id(parameter)) for name, parameter in model.named_parameters())
    with ExitStack() as stack:
        for index, block in enumerate(model.transformer.blocks):
            original = block.forward
            record = None if capture is None else capture.setdefault("blocks", {}).setdefault(str(index), {})

            def forward(x, *args, _original=original, _record=record, _index=index, _block=block, **kwargs):
                if _record is not None:
                    _record["forward_calls"] = _record.get("forward_calls", 0) + 1
                    remember_tensor(_record, "input", x)
                if _index in arm.fp32_attention_blocks:
                    result = ordinary_attention_fp32(_block, x, *args, **kwargs)
                elif _index in arm.fp32_blocks:
                    with torch.autocast(x.device.type, enabled=False):
                        result = _original(x.float(), *args, **kwargs)
                else:
                    result = _original(x, *args, **kwargs)
                if _record is not None:
                    remember_tensor(_record, "output", result[0])
                return result

            stack.enter_context(replace_method(block, "forward", forward))
            original_cast = block._cast_attn_bias
            sdpa_record = None if record is None else record.setdefault("sdpa", {})
            if record is not None or arm.fp32_bias:
                def cast(bias, input_dtype, _original=original_cast, _record=sdpa_record):
                    if _record is not None:
                        _record["bias_before_cast"] = cpu(bias)
                        _record["bias_before_cast_dtype"] = str(bias.dtype)
                    if arm.fp32_bias:
                        if bias.dtype != torch.float32:
                            raise AssertionError("ALiBi control expects the original FP32 bias")
                        return bias
                    return _original(bias, input_dtype)

                stack.enter_context(replace_method(block, "_cast_attn_bias", cast))
            if record is not None or arm.fp32_bias:
                original_sdpa = block._scaled_dot_product_attention

                def sdpa(q, k, v, *args, _original=original_sdpa, _record=sdpa_record, **kwargs):
                    if _record is not None:
                        _record["forward_calls"] = _record.get("forward_calls", 0) + 1
                        for name, value in (("q", q), ("k", k), ("v", v)):
                            remember_tensor(_record, name, value)
                        mask = kwargs.get("attn_mask", args[0] if args else None)
                        _record["mask"] = cpu(mask)
                        _record.setdefault("dtypes", {})["mask"] = None if mask is None else str(mask.dtype)
                        _record["is_causal"] = bool(kwargs.get("is_causal", False))
                        _record["dropout_p"] = float(kwargs.get("dropout_p", 0.0))
                        _record["autocast_enabled"] = torch.is_autocast_enabled(q.device.type)
                    if arm.fp32_bias:
                        # CUDA SDPA's autocast wrapper also rounds floating masks.
                        # Keep BF16 Q/K/V and output, disabling only that wrapper.
                        with torch.autocast(q.device.type, enabled=False):
                            output = _original(q, k, v, *args, **kwargs)
                        if q.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
                            raise AssertionError("ALiBi-only control must retain BF16 SDPA operands/output")
                    else:
                        output = _original(q, k, v, *args, **kwargs)
                    if _record is not None:
                        _record["operation_autocast_disabled_by_control"] = arm.fp32_bias
                        remember_tensor(_record, "output", output)
                    return output

                stack.enter_context(replace_method(block, "_scaled_dot_product_attention", sdpa))
        if arm.fp32_head:
            for module in (model.transformer.ln_f, model.transformer.ff_out):
                original = module.forward

                def final(x, _original=original):
                    with torch.autocast(x.device.type, enabled=False):
                        return _original(x.float())

                stack.enter_context(replace_method(module, "forward", final))
        yield
    if canonical != tuple((name, id(parameter)) for name, parameter in model.named_parameters()):
        raise AssertionError("Runtime wrappers changed canonical parameter ownership")


@contextmanager
def intervention(model, arm, capture=None):
    previous_lambda = model.config.cdrm_lambda
    previous_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        if arm.lambda_zero:
            model.config.cdrm_lambda = 0.0
        if arm.reduction_off:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        with ExitStack() as stack:
            stack.enter_context(ordinary_controls(model, arm, capture))
            if arm.bypass:
                def bypass(early, late, owner, attention_bias, output_states=False):
                    return late, {"p3": early, "p8": late} if output_states else None
                stack.enter_context(replace_method(model.cdrm, "forward", bypass))
            yield
    finally:
        model.config.cdrm_lambda = previous_lambda
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous_reduction


def bypass_packet(model, tokens, labels, mixed):
    """The bypass has no synthetic memory states or fabricated adapter gradients."""
    model.zero_grad(set_to_none=True)
    embeddings = model.transformer.wte(tokens)
    embeddings.retain_grad()
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
        output = model(None, input_embeddings=embeddings, output_cdrm_states=True)
        summed, count = loss_sum(output.logits, labels)
        loss = summed / count
    output.logits.retain_grad()
    for value in output.cdrm_states.values():
        value.retain_grad()
    loss.backward()
    packet = {"parameters": {name: cpu(value.grad) for name, value in model.named_parameters()},
              "inputs": {"embedding_output": cpu(embeddings.grad),
                         **{name: cpu(value.grad) for name, value in output.cdrm_states.items()}},
              "states": cpu(output.cdrm_states), "logits": cpu(output.logits),
              "cotangent": cpu(output.logits.grad), "loss": loss.item(), "native_targets": count,
              "input_gradient_scope": "Bypassed memory; only actual ordinary preview intermediates exist."}
    if sorted(name for name, value in packet["parameters"].items() if value is None) != sorted(ADAPTERS):
        raise AssertionError("Bypass must leave exactly the two adapters unused")
    for name, value in gradients(packet).items():
        if value is None and name in ADAPTERS:
            continue
        if value is None or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise AssertionError(f"Unexpected bypass gradient: {name}")
    return packet


def exact_tree(reference, actual, prefix=""):
    differences = []
    if torch.is_tensor(reference) or torch.is_tensor(actual):
        if not (torch.is_tensor(reference) and torch.is_tensor(actual)
                and reference.dtype == actual.dtype and torch.equal(reference, actual)):
            differences.append(prefix)
    elif isinstance(reference, dict) and isinstance(actual, dict):
        if set(reference) != set(actual):
            differences.append(prefix + "/keys")
        for key in reference.keys() & actual.keys():
            differences.extend(exact_tree(reference[key], actual[key], prefix + "/" + str(key)))
    elif isinstance(reference, (tuple, list)) and isinstance(actual, (tuple, list)):
        if len(reference) != len(actual):
            differences.append(prefix + "/length")
        for index, (r, a) in enumerate(zip(reference, actual)):
            differences.extend(exact_tree(r, a, prefix + "/" + str(index)))
    elif reference != actual:
        differences.append(prefix)
    return differences


def validate_capture(capture, count=5):
    if set(capture.get("blocks", {})) != {str(index) for index in range(count)}:
        raise AssertionError("Ordinary block capture coverage differs")
    for index, block in capture["blocks"].items():
        if block["forward_calls"] != 1 or block["sdpa"]["forward_calls"] != 1:
            raise AssertionError(f"Ordinary block {index} was not called exactly once")
        for record, names in ((block, ("input", "output")), (block["sdpa"], ("q", "k", "v", "output"))):
            for name in names:
                if record.get("gradient_hook_calls", {}).get(name) != 1:
                    raise AssertionError(f"Missing/duplicate block {index} {name} adjoint")
                if not bool(torch.isfinite(record[name]).all()) or not bool(torch.isfinite(record[name + "_gradient"]).all()):
                    raise AssertionError(f"Nonfinite block {index} {name} capture")


def restore_raw_gradients(model, packet, *, zero_unused_adapters=False):
    for name, parameter in model.named_parameters():
        value = packet["parameters"][name]
        if value is None:
            if name not in ADAPTERS:
                raise AssertionError(f"Unexpected absent gradient: {name}")
            parameter.grad = torch.zeros_like(parameter) if zero_unused_adapters else None
        else:
            parameter.grad = value.to(parameter.device).clone()


def run_arm(arm, checkpoint, tokens, labels, *, observed):
    started = time.monotonic()
    # Independent controls must not share Dynamo's bounded specialization list.
    # The retained Inductor disk cache remains active; no kernels/math change.
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    configure_compiled_helpers(True)
    net, construction = build_model("tiled", "bf16_fp32_state" if arm.bf16 else "fp32",
                                    0, checkpoint=checkpoint, eager=False)
    opt = optimizer_for(net)
    opt.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    initial_optimizer = adam_state(opt, net)
    hyperparameters = [{key: cpu(value) for key, value in group.items() if key != "params"}
                       for group in opt.param_groups]
    capture = {} if observed else None
    with intervention(net, arm, capture):
        packet = (bypass_packet(net, tokens, labels, arm.bf16) if arm.bypass
                  else full_packet(net, tokens, labels, arm.bf16, loss_sum))
    if observed:
        validate_capture(capture)
    step = step_packet(net, opt, 1.0)
    normalized_step = None
    if arm.bypass:
        net.load_state_dict(checkpoint["model"], strict=True)
        opt.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        restore_raw_gradients(net, packet, zero_unused_adapters=True)
        normalized_step = step_packet(net, opt, 1.0)
    result = {"packet": packet, "step": step, "normalized_zero_step": normalized_step,
              "capture": capture, "construction": construction,
              "initial_optimizer_named": initial_optimizer,
              "optimizer_hyperparameters": hyperparameters,
              "elapsed_seconds_including_capture": time.monotonic() - started}
    del net, opt
    return result


def subset_step(step, names):
    return {**step, **{key: {name: value for name, value in step[key].items() if name in names}
                      for key in ("deltas", "weights", "clipped_gradients", "state")}}


def record_compiler_audit(report, name, *, require_graphs):
    record = compiler_audit(False, require_graphs=False)
    record.update(required=True, require_graphs=require_graphs, passed=False)
    report.setdefault("compiler_by_arm", {})[name] = record
    compiler_audit(True, require_graphs=require_graphs)
    record["passed"] = True


def compiler_totals(records):
    totals = {}
    for record in records.values():
        for group, entries in record["counters"].items():
            for reason, count in entries.items():
                target = totals.setdefault(group, {})
                target[reason] = target.get(reason, 0) + count
    return {"scope": "Sum of separately retained per-arm audits before each Dynamo frame-cache reset; shared Inductor disk cache retained.",
            "audited_arms": len(records), "all_arms_pass": all(record["passed"] for record in records.values()),
            "failed_arms": [name for name, record in records.items() if not record["passed"]], "counters": totals}


def compare_arms(reference, actual, reference_step, actual_step, lr):
    names = [name for name, value in reference["parameters"].items() if value is not None]
    if names != [name for name, value in actual["parameters"].items() if value is not None]:
        raise AssertionError("Comparison requires matching gradient objectives/coverage")
    ref, act = gradients(reference), gradients(actual)
    ref = {name: value for name, value in ref.items() if value is not None}
    act = {name: value for name, value in act.items() if value is not None}
    return {"actual_ce_gradients": comparison(ref, act, names),
            "adam": adam_comparison(subset_step(reference_step, names), subset_step(actual_step, names),
                                    reference["parameters"], actual["parameters"], False, lr),
            "logits": metric(reference["logits"], actual["logits"]),
            "absolute_ce_difference": abs(actual["loss"] - reference["loss"]),
            "signed_ce_difference": actual["loss"] - reference["loss"],
            "parameter_scope": names}


def bypass_parity(zero, bypass, zero_step, bypass_step, normalized_step):
    shared = [name for name in zero["parameters"] if name not in ADAPTERS]
    values = {"loss": exact_tree(zero["loss"], bypass["loss"]),
              "logits": exact_tree(zero["logits"], bypass["logits"]),
              "shared_parameters": exact_tree({n: zero["parameters"][n] for n in shared},
                                               {n: bypass["parameters"][n] for n in shared}),
              "shared_inputs": exact_tree({n: zero["inputs"][n] for n in bypass["inputs"]}, bypass["inputs"]),
              "native_shared_step": exact_tree(subset_step(zero_step, shared), subset_step(bypass_step, shared)),
              "zero_normalized_full_step": exact_tree(zero_step, normalized_step)}
    return {"differences": values,
            "bitwise_shared_computation_and_normalized_step_equal": not any(
                value for name, value in values.items() if name != "native_shared_step"),
            "native_shared_step_equal": not values["native_shared_step"],
            "native_unused_gradients": {name: None if bypass["parameters"][name] is None else "present" for name in ADAPTERS},
            "scope": "Bypass adapters have native None gradients. The additional zero-normalized Adam packet explicitly recreates lambda-zero adapter updates; native bypass moments remain separately retained."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--arm", action="append", choices=tuple(ARMS), help="Repeat; default coarse suite. Baselines and matching references are automatic.")
    parser.add_argument("--verify-observer-arm", action="append", default=[], choices=tuple(ARMS),
                        help="Also repeat this selected control without observation and require bitwise packet/step parity.")
    parser.add_argument("--wandb-project", default="cdrm-numerical-resolution")
    parser.add_argument("--wandb-entity", default="taylorbollman")
    parser.add_argument("--wandb-group", default="20260907T232931Z")
    parser.add_argument("--wandb-run-name", default="fixed-state-precision-controls")
    args = parser.parse_args(argv)
    if not set(args.verify_observer_arm).issubset(selected_arms(args.arm)):
        parser.error("Observer-removal checks must name an explicitly selected control")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    tracker = None
    artifact = {"packets": {}, "steps": {}, "normalized_zero_steps": {}, "captures": {},
                "unobserved_baselines": {}, "unobserved_controls": {}}
    report = {"schema": "cdrm-fixed-state-precision-probe-v1", "status": "running", "numerical_clearance": False,
              "scope": "Saved same-state B64/T256/K96 diagnostics only; no training or acceptance-threshold revision.",
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "controls": [asdict(ARMS[name]) for name in selected_arms(args.arm)],
              "capture_scope": "Hooks record ordinary block boundaries and direct SDPA inputs/adjoints. They do not wrap shared child Linears or custom-scan replay.",
              "baseline_replay": {}, "observation_invariance": {}, "comparisons": {}, "bypass_parity": {}}
    try:
        report["runtime"] = setup(0)
        if digest(args.contract) != CONTRACT_SHA256:
            raise AssertionError("The prospective numerical-resolution contract differs")
        report["reference_contract"] = {"path": str(args.contract), "sha256": CONTRACT_SHA256}
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        source_report = json.loads((args.source_case / "report.json").read_text())
        if digest(args.checkpoint) != source_report["checkpoint"]["sha256"]:
            raise AssertionError("Checkpoint does not match the archived numerical role")
        if digest(args.source_case / "tensors.pt") != source_report["tensor_artifact"]["sha256"]:
            raise AssertionError("Archived numerical tensor checksum differs")
        if checkpoint["completed_updates"] != 1000:
            raise AssertionError("This bounded diagnostic expects the retained u1000 checkpoint")
        original = torch.load(args.source_case / "tensors.pt", map_location="cpu", weights_only=False)
        tokens, labels = original["tokens"].cuda(), original["labels"].cuda()
        if tuple(tokens.shape) != (64, 256) or labels.shape != tokens.shape or int((labels != -100).sum()) != 6144:
            raise AssertionError("Expected exactly the archived physical B64/T256/K96 fixture")
        artifact.update(tokens=cpu(tokens), labels=cpu(labels), initial_weights=cpu(checkpoint["model"]))
        tracked = sources((Path(__file__).resolve().relative_to(Path.cwd()),
                           Path("scripts/cdrm_tiled_validate.py"), Path("scripts/r3_backward_validate.py"),
                           Path("tests/test_cdrm_precision_probe.py"), args.contract))
        tracked.update(checkpoint["identity"]["source_sha256"])
        verify_sources(tracked)
        snapshot_sources(args.output_dir, tracked)
        report.update(source_sha256=tracked,
                      checkpoint={"path": str(args.checkpoint), "sha256": digest(args.checkpoint), "completed_updates": 1000},
                      source_case={"path": str(args.source_case), "report_sha256": digest(args.source_case / "report.json"),
                                   "tensor_sha256": digest(args.source_case / "tensors.pt"), "fixture": source_report["fixture"],
                                   "criteria": source_report["criteria"], "original_machine_screens_pass": source_report["machine_screens_pass"]})
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                                name=args.wandb_run_name, output_dir=args.output_dir, preserve_state=preserve_rng)
        report["wandb"] = tracker.record
        tracker.start({key: report[key] for key in ("scope", "controls", "checkpoint", "source_case", "source_sha256")})
        report["wandb"] = tracker.record
        print(json.dumps({"wandb_run_url": tracker.record["run_url"]}), flush=True)
        for name in dict.fromkeys((*BASELINES, *args.verify_observer_arm)):
            print(f"Unobserved replay: {name}", flush=True)
            result = run_arm(ARMS[name], checkpoint, tokens, labels, observed=False)
            target = "unobserved_baselines" if name in BASELINES else "unobserved_controls"
            artifact[target][name] = {"packet": result["packet"], "step": result["step"]}
            record_compiler_audit(report, "unobserved/" + name, require_graphs=True)
            if name not in BASELINES:
                continue
            old_name = "tiled_fp32" if name == "current_fp32" else "tiled_bf16"
            replay = {"packet_differences": exact_tree(original["packets"][old_name], result["packet"]),
                      "step_differences": exact_tree(original["steps"][old_name], result["step"])}
            replay["bitwise_equal"] = not replay["packet_differences"] and not replay["step_differences"]
            report["baseline_replay"][name] = replay
            if not replay["bitwise_equal"]:
                raise AssertionError(f"Unobserved {name} differs from archived arithmetic")
        for name in selected_arms(args.arm):
            print(f"Observed control: {name}", flush=True)
            result = run_arm(ARMS[name], checkpoint, tokens, labels, observed=True)
            for key, source in (("packets", "packet"), ("steps", "step"), ("captures", "capture")):
                artifact[key][name] = result[source]
            if result["normalized_zero_step"] is not None:
                artifact["normalized_zero_steps"][name] = result["normalized_zero_step"]
            record_compiler_audit(report, "observed/" + name, require_graphs=not ARMS[name].bypass)
            if "initial_optimizer_named" not in artifact:
                artifact["initial_optimizer_named"] = result["initial_optimizer_named"]
                artifact["optimizer_hyperparameters"] = result["optimizer_hyperparameters"]
            elif exact_tree(artifact["initial_optimizer_named"], result["initial_optimizer_named"]):
                raise AssertionError("Starting Adam state differs between controls")
            report.setdefault("construction", {})[name] = result["construction"]
            report.setdefault("capture_summary", {})[name] = {
                "elapsed_seconds_including_capture": result["elapsed_seconds_including_capture"],
                "blocks": {index: {"boundary_dtypes": block["dtypes"],
                                    "sdpa_dtypes": block["sdpa"]["dtypes"],
                                    "sdpa_bias_before_cast_dtype": block["sdpa"]["bias_before_cast_dtype"],
                                    "forward_calls": block["forward_calls"],
                                    "sdpa_gradient_hook_calls": block["sdpa"]["gradient_hook_calls"]}
                           for index, block in result["capture"]["blocks"].items()}}
            if name in BASELINES or name in args.verify_observer_arm:
                observed = {"packet": result["packet"], "step": result["step"]}
                target = "unobserved_baselines" if name in BASELINES else "unobserved_controls"
                differences = exact_tree(artifact[target][name], observed)
                report["observation_invariance"][name] = {"bitwise_equal": not differences, "differences": differences}
                if differences:
                    raise AssertionError(f"Observation changed {name} arithmetic")
            tracker.log({f"ce/{name}": result["packet"]["loss"], f"gradient_norm/{name}": result["step"]["clip_norm"]})
            print(json.dumps({"arm": name, "ce": result["packet"]["loss"], "clip_norm": result["step"]["clip_norm"]}), flush=True)
        lr = float(checkpoint["optimizer"]["param_groups"][0]["lr"])
        for name in selected_arms(args.arm):
            arm = ARMS[name]
            if not arm.bf16:
                continue
            reference = "fp32_bypass" if arm.bypass else "fp32_lambda_zero" if arm.lambda_zero else "current_fp32"
            result = compare_arms(artifact["packets"][reference], artifact["packets"][name],
                                  artifact["steps"][reference], artifact["steps"][name], lr)
            result["reference"] = reference
            report["comparisons"][name] = result
            tracker.log({f"gradient_relative_l2/{name}": result["actual_ce_gradients"]["global_parameter_relative_l2"],
                         f"adam_relative_l2/{name}": result["adam"]["global_delta_relative_l2"],
                         f"gradient_screen_pass/{name}": result["actual_ce_gradients"]["pass"],
                         f"adam_screen_pass/{name}": result["adam"]["guardrail_pass"]})
        for name in selected_arms(args.arm):
            if ARMS[name].bypass:
                zero = "bf16_lambda_zero" if ARMS[name].bf16 else "fp32_lambda_zero"
                report["bypass_parity"][name] = bypass_parity(
                    artifact["packets"][zero], artifact["packets"][name], artifact["steps"][zero],
                    artifact["steps"][name], artifact["normalized_zero_steps"][name])
        verify_sources(tracked)
        if digest(args.checkpoint) != report["checkpoint"]["sha256"] or digest(args.source_case / "report.json") != report["source_case"]["report_sha256"]:
            raise AssertionError("Archived scientific input changed during diagnosis")
        report["compiler"] = compiler_totals(report["compiler_by_arm"])
        report["status"] = "diagnostics_complete"
        tracker.summary({"diagnostics_complete": True, "numerical_clearance": False,
                         "archived_machine_screens_pass": source_report["machine_screens_pass"]})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if "compiler_by_arm" in report:
            report["compiler"] = compiler_totals(report["compiler_by_arm"])
        try:
            torch.save(artifact, args.output_dir / "tensors.pt")
            report["tensor_artifact"] = {"path": str(args.output_dir / "tensors.pt"),
                                         "sha256": digest(args.output_dir / "tensors.pt")}
            if tracker is not None:
                tracker.finish(succeeded=report["status"] == "diagnostics_complete")
        except Exception as error:
            report.update(status="failed", finalization_error_type=type(error).__name__)
            raise
        finally:
            save_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
